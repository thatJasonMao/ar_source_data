"""Canonical 268 trials, four information channels; legacy scope is untouched."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Guidance_Channel_Contract import CHANNELS, CONDITIONS, AVAILABLE

SCHEMA = 'a5_irl_four_channels_full_frames_v6_1'
COHORT_POLICY = 'all_conditions_exclude_ljn_textbar_four_channels_v6_1'
CHANNEL_COLUMNS = ['action', 'valid', 'previous_action', 'transition_valid',
                   'segment_id', 'run_id', 'run_weight', 'trap_observed']


def select_trials(canonical):
    trials = canonical.sort_values(['Scenario', 'ParticipantID', 'FileName']).reset_index(drop=True).copy()
    trials['trial_id'] = [f't{i:03}' for i in range(len(trials))]
    excluded = trials.ParticipantID.eq('LJN_C12') & trials.Guidance.eq('TextBar')
    return trials.loc[~excluded].copy(), trials.loc[excluded].copy()


def validate_manifest(manifest):
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete':
        raise ValueError('Require a completed V6 four-channel dataset')
    if manifest.get('cohort_policy') != COHORT_POLICY or manifest.get('channels') != CHANNELS:
        raise ValueError('V6 cohort/channel contract differs')
    if manifest.get('sampling_mode') != 'all_unique_subject_timestamps':
        raise ValueError('V6 requires original full-frame timestamps')
    from common import load_trials
    expected, _ = select_trials(load_trials())
    keys = {(r.trial_id, r.ParticipantID, r.Guidance, r.Smoke, r.FileName) for r in expected.itertuples()}
    actual = {(r['trial_id'], r['participant_id'], r['condition'], r['smoke'], r['source_file']) for r in manifest['trials']}
    if actual != keys or len(actual) != len(manifest['trials']) or manifest['cohort_trials'] != len(keys):
        raise ValueError('V6 manifest does not match canonical trial identities')
    for r in manifest['trials']:
        if r['channels'] != list(AVAILABLE[r['condition']]):
            raise ValueError('Incorrect channel availability')
        if r['split'] != manifest['participant_splits'][r['participant_id']]:
            raise ValueError('Participant split mismatch')
