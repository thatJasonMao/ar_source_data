"""Task-local IRL scope, applied after the canonical common.load_trials cohort."""
from __future__ import annotations

SCHEMA = 'a5_irl_inputs_v2'
COHORT_POLICY = 'ar_only_exclude_ljn_textbar_20260927'
GUIDANCE = ['ArrowRibbon', 'MiniMap', 'TextBar', 'Full']


def exclusion_reason(participant_id, guidance):
    if guidance == 'StaticSign':
        return 'staticsign_outside_irl_scope'
    if guidance not in GUIDANCE:
        return 'unsupported_guidance'
    if participant_id == 'LJN_C12' and guidance == 'TextBar':
        return 'ljn_c12_textbar_excluded_by_user'
    return ''


def select_irl_trials(canonical):
    """Keep stable IDs from the full cohort; never scan raw directories here."""
    trials = canonical.sort_values(['Scenario', 'ParticipantID', 'FileName']).reset_index(drop=True).copy()
    trials['trial_id'] = [f't{i:03}' for i in range(len(trials))]
    reasons = trials.apply(lambda r: exclusion_reason(r.ParticipantID, r.Guidance), axis=1)
    included = trials.loc[reasons.eq('')].copy()
    excluded = trials.loc[reasons.ne('')].copy()
    excluded['exclusion_reason'] = reasons.loc[excluded.index]
    return included.reset_index(drop=True), excluded.reset_index(drop=True)


def validate_manifest_scope(manifest):
    """Reject old/excluded cohorts before any split is read, fitted or exported."""
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete':
        raise ValueError('Require a completed '+SCHEMA+' dataset; rebuild older inputs')
    if manifest.get('cohort_policy') != COHORT_POLICY:
        raise ValueError('Dataset does not implement the confirmed IRL cohort policy')
    records = manifest['trials']
    if len(records) != manifest['cohort_trials']:
        raise ValueError('Manifest cohort count differs from trial records')
    for record in records:
        reason = exclusion_reason(record['participant_id'], record['guidance'])
        if reason:
            raise ValueError('Excluded IRL trial: '+record['trial_id']+' ('+reason+')')
