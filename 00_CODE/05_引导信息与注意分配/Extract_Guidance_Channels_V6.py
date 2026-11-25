"""Extract StaticSign in all canonical trials and integrate four AOI channels.

Legacy AR AOI geometry and UI-over-carpet precedence are reused unchanged.
StaticSign uses direct logged Hit1 names. Multi-label overlaps are retained and
audited, not counted as separate events in AnyGuidanceGaze.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from common import load_trials
from Reconstruct_LightBlanket_Gaze_AOI import read_csv_safely
from Guidance_Channel_Contract import (ROOT, CHANNELS, AVAILABLE, FIELDS, AOI_ROOT,
    AOI_SCHEMA, aoi_path, timestamp_ns, static_arrays, as_bool, digest)


def integrate(row, legacy_root):
    eye = read_csv_safely(Path(row.EyeGazeFile))
    ns = timestamp_ns(eye.TimeStamp)
    subject = read_csv_safely(Path(row.SubjectFile))
    if not np.array_equal(ns, timestamp_ns(subject.TimeStamp)):
        raise ValueError('Subject/EyeGaze frame alignment differs: ' + row.FileName)
    trial = row.FileName.split('_Subject_Simulation_Info_', 1)[0]
    out = eye[['TimeStamp', 'Hit1_X', 'Hit1_Y', 'Hit1_Z', 'Hit1_Name']].copy()
    out.insert(0, 'OriginalRowIndex', np.arange(len(out)))
    for name, value in dict(Condition=row.Scenario, GuidanceType=row.Guidance,
            SmokeLevel=row.Smoke, ParticipantFolder=row.ParticipantID, TrialID=trial).items():
        out[name] = value
    valid, hit = static_arrays(eye)
    out['ValidStaticSignAOI'] = valid.astype('int8')
    out['GazeOnStaticSign'] = hit.astype('int8')
    out['GazeOnStaticSignRaw'] = hit.astype('int8')
    out['StaticSignAOIStatus'] = np.where(~valid, 'invalid_eye_hit', np.where(hit, 'on_static_sign', 'off_static_sign'))
    sources = [Path(row.SubjectFile), Path(row.EyeGazeFile)]
    old = None
    if row.Guidance != 'StaticSign':
        path = aoi_path(row, legacy_root)
        old = pd.read_csv(path, low_memory=False)
        if not np.array_equal(ns, timestamp_ns(old.TimeStamp)):
            raise ValueError('Legacy AOI timestamp mismatch: ' + str(path))
        if not np.array_equal(old.OriginalRowIndex.to_numpy(), out.OriginalRowIndex.to_numpy()):
            raise ValueError('Legacy original row index differs: ' + str(path))
        sources.append(path)
    valids, hits = [], []
    for channel in CHANNELS:
        v, g = FIELDS[channel]
        available = channel in AVAILABLE[row.Guidance]
        out['Available' + channel] = int(available)
        if channel != 'StaticSign':
            out[v] = as_bool(old[v]).astype('int8') if available else np.zeros(len(out), np.int8)
            out[g] = (as_bool(old[g]) & as_bool(old[v])).astype('int8') if available else np.zeros(len(out), np.int8)
        valids.append(as_bool(out[v]))
        hits.append(as_bool(out[g]) & valids[-1])
    out['ValidAnyGuidanceAOI'] = np.logical_or.reduce(valids).astype('int8')
    out['AnyGuidanceGaze'] = np.logical_or.reduce(hits).astype('int8')
    out['GuideMatchCount'] = np.sum(hits, axis=0).astype('int8')
    ui = hits[2] | hits[3]
    out['StaticSignUIOverlap'] = (hit & ui).astype('int8')
    out['GazeOnStaticSignUIExclusive'] = (hit & ~ui).astype('int8')
    codes = np.sum([h.astype(int) * (1 << i) for i, h in enumerate(hits)], axis=0)
    labels = {i: '+'.join(c for j, c in enumerate(CHANNELS) if i & (1 << j)) or 'None' for i in range(16)}
    out['GuidanceAOILabel'] = pd.Series(codes).map(labels).to_numpy()
    out.loc[out.ValidAnyGuidanceAOI.eq(0), 'GuidanceAOILabel'] = 'Unavailable'
    return out, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=AOI_ROOT)
    p.add_argument('--legacy-root', type=Path, default=ROOT / 'Results_Guidance_GazeAOI_Integrated')
    args = p.parse_args()
    outroot = args.output.resolve()
    outroot.mkdir(parents=True, exist_ok=False)
    roster = load_trials().sort_values(['Scenario', 'ParticipantID', 'FileName']).reset_index(drop=True)
    manifest = dict(schema=AOI_SCHEMA, status='building', created_at=datetime.now().isoformat(),
        channels=CHANNELS, available=AVAILABLE, cohort_trials=len(roster), cohort_participants=roster.ParticipantID.nunique(),
        static_names=['ABCD_Signal'] + [f'ABCD_Signal ({i})' for i in range(1, 14)],
        static_rule='valid Hit1 and exact name; no duration filtering; raw overlaps retained',
        validity='per-channel validity; union validity is OR of available valid channels',
        source_sha256={str(Path(f).resolve()): digest(f) for f in [__file__, ROOT / 'Guidance_Channel_Contract.py', ROOT / 'common.py', ROOT / 'Reconstruct_LightBlanket_Gaze_AOI.py']}, trials=[])
    save = lambda: (outroot / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    audit = []
    for row in tqdm(roster.itertuples(), total=len(roster), desc='Extract four guidance channels'):
        frame, inputs = integrate(row, args.legacy_root)
        target = aoi_path(row, outroot)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False, encoding='utf-8-sig')
        for channel in AVAILABLE[row.Guidance]:
            v, g = FIELDS[channel]
            audit.append(dict(scenario=row.Scenario, condition=row.Guidance, participant=row.ParticipantID,
                source_file=row.FileName, channel=channel, rows=len(frame), valid=int(frame[v].sum()),
                hits=int(frame[g].sum()), static_ui_overlap=int(frame.StaticSignUIOverlap.sum())))
        manifest['trials'].append(dict(scenario=row.Scenario, participant=row.ParticipantID,
            source_file=row.FileName, file=str(target.relative_to(outroot)), rows=len(frame), sha256=digest(target),
            input_hashes={str(p.resolve()): digest(p) for p in inputs}))
    audit = pd.DataFrame(audit)
    audit.to_csv(outroot / 'trial_channel_audit.csv', index=False)
    audit.groupby(['condition', 'channel']).agg(trials=('rows', 'size'), rows=('rows', 'sum'), valid=('valid', 'sum'), hits=('hits', 'sum')).to_csv(outroot / 'condition_channel_summary.csv')
    manifest.update(status='complete', rows=sum(r['rows'] for r in manifest['trials']), trained=False)
    save()
    print('OUTPUT=' + str(outroot))


if __name__ == '__main__':
    main()
