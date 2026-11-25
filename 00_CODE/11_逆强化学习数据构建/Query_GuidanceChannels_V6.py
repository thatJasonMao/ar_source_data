"""Exact or bounded previous-sample queries for each V6 information channel."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Guidance_Channel_Contract import AOI_ROOT, AVAILABLE, channel_actions, aoi_path, timestamp_ns
from Query_GuidanceGaze import event_flags


class GuidanceChannelQuery:
    def __init__(self, aoi_root=AOI_ROOT):
        self.root = Path(aoi_root)
        self.cache = {}

    def load(self, row):
        path = aoi_path(row, self.root)
        if path not in self.cache:
            frame = pd.read_csv(path, low_memory=False)
            self.cache[path] = (frame, timestamp_ns(frame.TimeStamp))
        return self.cache[path]

    def query_frames(self, row, channel, times_ns, *, match='exact', max_age_ms=50., events=False):
        if match not in ('exact', 'previous') or not np.isfinite(max_age_ms) or max_age_ms < 0:
            raise ValueError('Invalid matching options')
        frame, ns = self.load(row)
        source = channel_actions(frame, row.Guidance, channel)
        if events:
            flags = event_flags(ns, source == 1)[0]
            source = np.where(source >= 0, flags.astype(np.int8), -1)
        query = np.asarray(times_ns, np.int64)
        indices = np.searchsorted(ns, query, side='right') - 1
        safe = np.clip(indices, 0, len(ns) - 1)
        age = (query - ns[safe]) / 1e6
        matched = (indices >= 0) & (query <= ns[-1]) & (age >= 0) & (age <= max_age_ms)
        if match == 'exact':
            matched &= age == 0
        return np.where(matched, source[safe], -1).astype(np.int8)

    def clear_cache(self):
        self.cache.clear()


class StaticFrameAdapter:
    """Supply genuine StaticSign labels to the unchanged V3 feature builder."""
    def __init__(self, query, row):
        self.query, self.row = query, row

    def query_frame_many(self, participant_id, timestamps, *, scenario, match, max_age_ms):
        if participant_id != self.row.ParticipantID or scenario != self.row.Scenario:
            raise ValueError('Mismatched trial')
        times = np.array([pd.Timestamp(t).value for t in timestamps], dtype=np.int64)
        actions = self.query.query_frames(self.row, 'StaticSign', times, match=match, max_age_ms=max_age_ms)
        return [SimpleNamespace(status='ok' if a >= 0 else 'invalid_aoi', is_gazing=bool(a == 1),
            sample_age_ms=0. if a >= 0 else None, guidance_types=('StaticSign',) if a == 1 else ()) for a in actions]
