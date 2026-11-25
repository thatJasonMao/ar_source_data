"""Participant/time lookup of recorded guidance AOI gaze for RL replay.

No images, floor filtering, spatial smoothing or group normalization are used.
The default query accepts only continuous >=100 ms events (no singletons).
The separate frame query returns the matched frame's valid AOI hit directly.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIME_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"
MIN_EVENT_NS = 100_000_000
MAX_GAP_NS = 100_000_000
CHANNELS = {
    "ArrowRibbon": (("LightBlanket", "ValidLightBlanketAOI", "GazeOnLightBlanket"),),
    "MiniMap": (("MiniMap", "ValidMiniMapAOI", "GazeOnMiniMap"),),
    "TextBar": (("Text", "ValidTextAOI", "GazeOnTextGuidance"),),
}
CHANNELS["Full"] = sum(CHANNELS.values(), ())


def _timestamp(value) -> pd.Timestamp:
    if isinstance(value, str):
        if not re.match(r"^\d{4}[_-]\d{2}[_-]\d{2}[_ T]\d{2}[_:]\d{2}[_:]\d{2}", value):
            raise ValueError("Timestamp must include an explicit date and time to seconds")
        result = pd.to_datetime(value, format=TIME_FORMAT) if "_" in value else pd.Timestamp(value)
    elif isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
        result = pd.Timestamp(value)
    else:
        raise ValueError("Use a full recorded/ISO timestamp or datetime, not numeric elapsed/Unix time")
    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError("Timestamp must be finite and timezone-naive, in the experiment clock")
    return result


def _bool(series: pd.Series) -> np.ndarray:
    """Match integration/heatmap boolean conventions, reject unknown encodings."""
    text = series.fillna(0).astype(str).str.strip().str.lower()
    accepted = {"true", "false", "1", "0", "1.0", "0.0", "yes", "no", "y", "n", ""}
    if not text.isin(accepted).all():
        raise ValueError(f"Invalid boolean values in {series.name}")
    return text.isin({"true", "1", "1.0", "yes", "y"}).to_numpy(bool)


def sample_weights_ns(ns: np.ndarray) -> np.ndarray:
    """Forward intervals capped at 100 ms, with a rounded median terminal interval.

    Shared by event labelling and condition-level duration statistics. Fewer
    than two timestamps provide no estimable interval and receive zero weight.
    """
    ns = np.asarray(ns, dtype=np.int64)
    if ns.ndim != 1 or np.any(np.diff(ns) <= 0):
        raise ValueError("Sample times must be a strictly increasing 1D array")
    if len(ns) < 2:
        return np.zeros(len(ns), dtype=np.int64)
    delta = np.diff(ns)
    median = int(round(float(np.median(delta))))
    return np.minimum(np.r_[delta, median], MAX_GAP_NS)


def event_flags(ns: np.ndarray, hits: np.ndarray):
    """Label entire >=100 ms runs, rejecting singleton hits unconditionally.

    False/invalid samples or >100 ms gaps break a run. As in the heatmap,
    duration sums forward sample intervals capped at 100 ms; the final sample
    uses the median positive interval. Integer-nanosecond arithmetic avoids
    rejecting exactly-100 ms events due to floating-point rounding.
    Returns retained flags, event durations (ms), first/last hit indices.
    """
    ns = np.asarray(ns, dtype=np.int64)
    hits = np.asarray(hits, dtype=bool)
    if ns.ndim != 1 or hits.shape != ns.shape or np.any(np.diff(ns) <= 0):
        raise ValueError("Event samples require equal 1D arrays and strictly increasing times")
    retained = np.zeros(len(ns), dtype=bool)
    duration_ms = np.zeros(len(ns), dtype=float)
    first = np.full(len(ns), -1, dtype=np.int64)
    last = first.copy()
    if len(ns) < 2:
        return retained, duration_ms, first, last
    weights = sample_weights_ns(ns)
    indices = np.flatnonzero(hits)
    breaks = np.flatnonzero((np.diff(indices) > 1) |
                           (np.diff(ns[indices]) > MAX_GAP_NS)) + 1
    for event in np.split(indices, breaks):
        if len(event) < 2:
            continue
        duration = int(weights[event].sum())
        if duration >= MIN_EVENT_NS:
            retained[event] = True
            duration_ms[event] = duration / 1e6
            first[event] = event[0]
            last[event] = event[-1]
    return retained, duration_ms, first, last


@dataclass(frozen=True)
class GuidanceGazeResult:
    status: str
    participant_id: str
    query_timestamp: str
    is_gazing: bool | None = None
    guidance_types: tuple[str, ...] = ()
    scenario: str | None = None
    trial_file: str | None = None
    matched_timestamp: str | None = None
    sample_age_ms: float | None = None
    event_duration_ms: float | None = None
    event_start_timestamp: str | None = None
    event_last_hit_timestamp: str | None = None
    source_file: str | None = None
    message: str = ""


class GuidanceGazeQuery:
    """Lazy cached lookup. Construct once and reuse across RL steps.

    Default roster is common.load_trials() (45 participants / 270 trials).
    An explicit trials DataFrame allows a caller-controlled/test roster; it
    must contain Scenario, Guidance, ParticipantID, SubjectFolder, FileName.
    Snapshot caches are refreshed only by clear_cache() or a new instance.
    """
    def __init__(self, project_root=PROJECT_ROOT, *, trials=None):
        self.root = Path(project_root).resolve()
        if trials is None:
            spec = importlib.util.spec_from_file_location("_guidance_gaze_common", self.root / "common.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            trials = module.load_trials()
        required = {"Scenario", "Guidance", "ParticipantID", "SubjectFolder", "FileName"}
        if not required.issubset(trials.columns):
            raise ValueError(f"Trial roster requires {sorted(required)}")
        self.trials = trials.copy()
        self._participants = frozenset(self.trials.ParticipantID)
        if self.trials.duplicated(["Scenario", "ParticipantID", "FileName"]).any():
            raise ValueError("Duplicate trial identity in roster")
        self.clear_cache()

    def clear_cache(self):
        """Clear cached bounds/labels after regenerating source CSVs."""
        self._indices = {}
        self._frames = {}

    def _participant_index(self, participant):
        if participant not in self._indices:
            index = []
            rows = self.trials[self.trials.ParticipantID.eq(participant)]
            for row in rows.itertuples(index=False):
                subject = self.root / "RAW_DATA" / row.Scenario / row.SubjectFolder / row.FileName
                stamps = pd.read_csv(subject, usecols=["TimeStamp"])["TimeStamp"]
                parsed = pd.to_datetime(stamps.astype(str), format=TIME_FORMAT, errors="coerce").dropna()
                if parsed.empty:
                    raise ValueError(f"No valid Subject timestamps: {subject}")
                index.append((parsed.min().value, parsed.max().value, row))
            self._indices[participant] = index
        return self._indices[participant]

    def _aoi_path(self, row):
        marker = "_Subject_Simulation_Info_"
        if marker not in row.FileName:
            raise ValueError(f"Unrecognized Subject file: {row.FileName}")
        trial, stamp = row.FileName.split(marker, 1)
        return (self.root / "Results_Guidance_GazeAOI_Integrated" /
                row.Scenario / row.ParticipantID / f"{trial}_GuidanceGazeIntegrated_{stamp}")

    def _load_frame(self, row):
        path = self._aoi_path(row)
        if path in self._frames:
            return self._frames[path]
        channels = CHANNELS[row.Guidance]
        columns = ["TimeStamp", *[c for _, valid, gaze in channels for c in (valid, gaze)]]
        if row.Guidance == "Full":
            columns.append("AnyGuidanceGaze")
        frame = pd.read_csv(path, usecols=columns, low_memory=False)
        parsed = pd.to_datetime(frame.TimeStamp.astype(str), format=TIME_FORMAT, errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"Invalid AOI timestamps: {path}")
        frame = frame.assign(_time=parsed).sort_values("_time", kind="stable")
        if frame._time.duplicated().any():
            raise ValueError(f"Duplicate AOI timestamps: {path}; refusing ambiguous samples")
        if frame.empty:
            raise ValueError(f"Empty AOI data: {path}")
        ns = frame._time.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        valid = np.logical_or.reduce([_bool(frame[v]) for _, v, _ in channels])
        gaze_column = "AnyGuidanceGaze" if row.Guidance == "Full" else channels[0][2]
        hits = _bool(frame[gaze_column]) & valid
        types = tuple((name, _bool(frame[gaze]) & _bool(frame[v])) for name, v, gaze in channels)
        retained, duration_ms, first, last = event_flags(ns, hits)
        data = dict(ns=ns, valid=valid, hits=hits, retained=retained, duration_ms=duration_ms,
                    first=first, last=last, types=types, path=path)
        self._frames[path] = data
        return data

    def query(self, participant_id, timestamp, *, scenario=None,
              match="previous", max_age_ms=50.0) -> GuidanceGazeResult:
        """Query an absolute experiment timestamp, matching past samples only.

        Only >=100 ms multi-sample events count. match='exact' disallows fallback;
        'previous' uses the latest sample at/before time, at most max_age_ms old.
        Unknown/unavailable is None, never False. No matching across trials.
        """
        return self._query(participant_id, timestamp, scenario=scenario,
                           match=match, max_age_ms=max_age_ms, frame_only=False)

    def query_frame(self, participant_id, timestamp, *, scenario=None,
                    match="previous", max_age_ms=50.0) -> GuidanceGazeResult:
        """Return the matched frame's AOI hit, without duration filtering.

        Single-frame hits count. Alignment and missing-value rules match query().
        Use match='exact' to require the original frame timestamp. Event fields
        remain None: this result does not describe a sustained fixation event.
        This reads recorded AOI data, not a live gaze reconstruction.
        """
        return self._query(participant_id, timestamp, scenario=scenario,
                           match=match, max_age_ms=max_age_ms, frame_only=True)

    def _query(self, participant_id, timestamp, *, scenario, match,
               max_age_ms, frame_only) -> GuidanceGazeResult:
        if match not in {"previous", "exact"}:
            raise ValueError("match must be previous or exact")
        if not np.isfinite(max_age_ms) or max_age_ms < 0:
            raise ValueError("max_age_ms must be finite and nonnegative")
        stamp = _timestamp(timestamp)
        base = dict(participant_id=str(participant_id), query_timestamp=stamp.isoformat())
        if participant_id not in self._participants:
            return GuidanceGazeResult("unknown_participant", **base)
        try:
            candidates = [row for lo, hi, row in self._participant_index(participant_id)
                          if lo <= stamp.value <= hi and (scenario is None or row.Scenario == scenario)]
        except (OSError, ValueError, KeyError) as exc:
            return GuidanceGazeResult("invalid_source", **base, message=str(exc))
        if not candidates:
            return GuidanceGazeResult("outside_trial", **base, scenario=scenario)
        if len(candidates) > 1:
            return GuidanceGazeResult("ambiguous_trial", **base,
                                      message="Overlapping trial times; supply scenario or correct the roster")
        row = candidates[0]
        base.update(scenario=row.Scenario, trial_file=row.FileName)
        if row.Guidance not in CHANNELS:
            return GuidanceGazeResult("unsupported_guidance", **base,
                                      message="No comparable guidance AOI for this condition (e.g. StaticSign)")
        try:
            data = self._load_frame(row)
        except FileNotFoundError as exc:
            return GuidanceGazeResult("missing_aoi", **base, message=str(exc))
        except (OSError, ValueError, KeyError) as exc:
            return GuidanceGazeResult("invalid_source", **base, message=str(exc))
        base["source_file"] = str(data["path"])
        ns = data["ns"]
        if stamp.value < ns[0] or stamp.value > ns[-1]:
            return GuidanceGazeResult("outside_aoi_range", **base)
        i = int(np.searchsorted(ns, stamp.value, side="right")) - 1
        age = (stamp.value - int(ns[i])) / 1e6
        base.update(matched_timestamp=pd.Timestamp(int(ns[i])).isoformat(), sample_age_ms=age)
        if (match == "exact" and age != 0) or age > max_age_ms:
            return GuidanceGazeResult("no_sample", **base)
        if not data["valid"][i]:
            return GuidanceGazeResult("invalid_aoi", **base)
        if frame_only:
            hit = bool(data["hits"][i])
            return GuidanceGazeResult("ok", **base, is_gazing=hit,
                guidance_types=tuple(name for name, flags in data["types"] if hit and flags[i]))
        retained = bool(data["retained"][i])
        return GuidanceGazeResult("ok", **base, is_gazing=retained,
            guidance_types=tuple(name for name, flags in data["types"] if retained and flags[i]),
            event_duration_ms=float(data["duration_ms"][i]),
            event_start_timestamp=pd.Timestamp(int(ns[data["first"][i]])).isoformat() if retained else None,
            event_last_hit_timestamp=pd.Timestamp(int(ns[data["last"][i]])).isoformat() if retained else None)

    def query_many(self, participant_id, timestamps, **kwargs):
        """Return results in input order, reusing trial and AOI caches."""
        return [self.query(participant_id, stamp, **kwargs) for stamp in timestamps]

    def query_frame_many(self, participant_id, timestamps, **kwargs):
        """Batch frame-hit queries, in input order, sharing existing caches."""
        return [self.query_frame(participant_id, stamp, **kwargs) for stamp in timestamps]


@lru_cache(maxsize=1)
def get_engine() -> GuidanceGazeQuery:
    return GuidanceGazeQuery()


def query_guidance_gaze(participant_id, timestamp, **kwargs) -> GuidanceGazeResult:
    return get_engine().query(participant_id, timestamp, **kwargs)


def is_gazing_guidance(participant_id, timestamp, **kwargs) -> bool | None:
    """True=gazing, False=valid non-gazing, None=unknown; never bool(None)."""
    return query_guidance_gaze(participant_id, timestamp, **kwargs).is_gazing


def query_guidance_gaze_frame(participant_id, timestamp, **kwargs) -> GuidanceGazeResult:
    """Detailed single-frame AOI query; no minimum fixation duration."""
    return get_engine().query_frame(participant_id, timestamp, **kwargs)


def is_gazing_guidance_frame(participant_id, timestamp, **kwargs) -> bool | None:
    """True=valid frame hit, False=valid non-hit, None=unknown."""
    return query_guidance_gaze_frame(participant_id, timestamp, **kwargs).is_gazing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("participant_id")
    parser.add_argument("timestamp")
    parser.add_argument("--scenario")
    parser.add_argument("--match", choices=["previous", "exact"], default="previous")
    parser.add_argument("--max-age-ms", type=float, default=50.)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--frame", action="store_true",
                        help="Return the matched frame's AOI hit without the 100 ms event filter")
    args = parser.parse_args()
    engine = GuidanceGazeQuery(args.project_root)
    query = engine.query_frame if args.frame else engine.query
    result = query(args.participant_id, args.timestamp,
        scenario=args.scenario, match=args.match, max_age_ms=args.max_age_ms)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
