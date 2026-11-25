"""Four information channels across five experimental conditions (V6).

Names in CHANNELS identify information, never the experimental condition.
Raw per-channel hits can overlap; their Boolean union counts each frame once.
"""
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CHANNELS = ['StaticSign', 'ArrowRibbon', 'MiniMap', 'TextBar']
CONDITIONS = ['StaticSign', 'ArrowRibbon', 'MiniMap', 'TextBar', 'Full']
# Primary condition analysis. AVAILABLE remains the historical availability
# contract and must not be used to pool channels in experimental-mode figures.
CONDITION_CHANNELS = {
    'StaticSign': ('StaticSign',),
    'ArrowRibbon': ('ArrowRibbon',),
    'MiniMap': ('MiniMap',),
    'TextBar': ('TextBar',),
    'Full': ('ArrowRibbon', 'MiniMap', 'TextBar'),
}
CONDITION_LABELS = dict(zip(CONDITIONS, ['StaticSign', 'Arrow', 'MiniMap', 'TextBar', 'Multimodal']))
AVAILABLE = {
    'StaticSign': ('StaticSign',),
    'ArrowRibbon': ('StaticSign', 'ArrowRibbon'),
    'MiniMap': ('StaticSign', 'MiniMap'),
    'TextBar': ('StaticSign', 'TextBar'),
    'Full': tuple(CHANNELS),
}
FIELDS = {
    'StaticSign': ('ValidStaticSignAOI', 'GazeOnStaticSign'),
    'ArrowRibbon': ('ValidLightBlanketAOI', 'GazeOnLightBlanket'),
    'MiniMap': ('ValidMiniMapAOI', 'GazeOnMiniMap'),
    'TextBar': ('ValidTextAOI', 'GazeOnTextGuidance'),
}
AOI_SCHEMA = 'a5_guidance_channels_v6_1'
AOI_ROOT = ROOT / 'Results_Guidance_GazeAOI_Channels_V6'
TIME_FORMAT = '%Y_%m_%d_%H_%M_%S_%f'
STATIC_PATTERN = re.compile(r'ABCD_Signal(?: \((?:[1-9]|1[0-3])\))?')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def as_bool(s):
    text = s.fillna(0).astype(str).str.strip().str.lower()
    if not text.isin(['true', 'false', '1', '0', '1.0', '0.0', 'yes', 'no', 'y', 'n', '']).all():
        raise ValueError('Invalid Boolean encoding: ' + str(s.name))
    return text.isin(['true', '1', '1.0', 'yes', 'y']).to_numpy()


def aoi_path(row, root=AOI_ROOT):
    trial, stamp = row.FileName.split('_Subject_Simulation_Info_', 1)
    return Path(root) / row.Scenario / row.ParticipantID / f'{trial}_GuidanceGazeIntegrated_{stamp}'


def timestamp_ns(s):
    time = pd.to_datetime(s.astype(str), format=TIME_FORMAT, errors='raise')
    ns = time.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    if time.isna().any() or np.any(np.diff(ns) <= 0):
        raise ValueError('AOI timestamps must be valid, unique and increasing')
    return ns


def static_arrays(eye):
    names = eye.Hit1_Name.fillna('').astype(str).str.strip()
    xyz = eye[['Hit1_X', 'Hit1_Y', 'Hit1_Z']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    valid = (np.isfinite(xyz).all(1) & ~np.isclose(xyz, 0).all(1)
             & ~names.isin(['', 'Null', 'nan', 'None']).to_numpy())
    hit = names.str.fullmatch(STATIC_PATTERN).to_numpy() & valid
    return valid, hit


def channel_actions(frame, condition, channel):
    """-1 is unavailable/invalid; zero is an observed, valid negative."""
    if channel not in CHANNELS:
        raise ValueError('Unknown channel: ' + channel)
    if channel not in AVAILABLE[condition]:
        return np.full(len(frame), -1, np.int8)
    v, g = FIELDS[channel]
    return np.where(as_bool(frame[v]), as_bool(frame[g]).astype(np.int8), -1).astype(np.int8)


def condition_gaze_masks(frame, condition, ns, event_labeler):
    """Filter each condition's own channels before union; never sum overlaps.

    The supplied labeler is Query_GuidanceGaze.event_flags: this helper adds
    no event-recognition method. Validity is independent of event filtering.
    """
    channels = CONDITION_CHANNELS[condition]
    actions = [channel_actions(frame, condition, channel) for channel in channels]
    raw_channels = np.stack([action == 1 for action in actions])
    valid = np.logical_or.reduce([action >= 0 for action in actions])
    labelled = [event_labeler(ns, hit) for hit in raw_channels]
    sustained_channels = np.stack([result[0] for result in labelled])
    raw = raw_channels.any(axis=0)
    sustained = sustained_channels.any(axis=0)
    if np.any(sustained & ~raw) or np.any(raw & ~valid):
        raise ValueError('Condition masks violate numerator/denominator inclusion')
    return dict(channels=channels, valid=valid, raw=raw, sustained=sustained,
                raw_channels=raw_channels, sustained_channels=sustained_channels,
                channel_event_counts={channel: int(len(np.unique(result[2][result[0]])))
                                      for channel, result in zip(channels, labelled)})
