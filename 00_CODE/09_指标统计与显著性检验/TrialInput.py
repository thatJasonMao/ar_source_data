"""Strict metric-table alignment to common.load_trials(), shared by invokers."""
import numpy as np
import pandas as pd


def prepare_trial_data(raw, common, metric):
    """Require exactly one finite metric for each canonical trial; never infer IDs."""
    trials = common.load_trials()
    keys = ['Scenario', 'ParticipantID', 'FileName']
    required = {'Scenario', 'FileName', metric}
    if not required.issubset(raw.columns) or not {'Subject', 'ParticipantID'} & set(raw.columns):
        raise ValueError(f'CSV requires {sorted(required)} and Subject or ParticipantID')
    audit = raw.copy().reset_index(drop=True)
    audit['SourceRow'] = np.arange(2, len(audit) + 2)
    if 'ParticipantID' not in audit:
        audit['ParticipantID'] = audit['Subject']
    if 'Subject' not in audit:
        audit['Subject'] = audit['ParticipantID']
    if audit[keys + ['Subject']].isna().any().any():
        raise ValueError('Missing trial identity in source CSV')
    if not audit.Subject.eq(audit.ParticipantID).all():
        raise ValueError('Subject and ParticipantID conflict; session suffixes are not inferred')
    if audit.duplicated(keys).any():
        raise ValueError('Duplicate trial rows in source CSV; regenerate the upstream table')
    expected = set(trials[keys].itertuples(index=False, name=None))
    actual = set(audit[keys].itertuples(index=False, name=None))
    if len(trials) != 270 or trials.ParticipantID.nunique() != 45:
        raise ValueError('common.load_trials() did not return 45 participants / 270 trials')
    if len(audit) != 270 or actual != expected:
        raise ValueError(
            f'Source CSV must match common.load_trials() exactly: rows={len(audit)}, '
            f'missing={len(expected - actual)}, outside_cohort={len(actual - expected)}; '
            'regenerate the upstream table')
    # All grouping columns are taken from the authoritative interface.
    grouping = trials[keys + ['Group', 'Guidance', 'Smoke']]
    aligned = audit[keys].merge(grouping, on=keys, how='left', validate='one_to_one', sort=False)
    for source_col, canonical_col in [('Group', 'Group'), ('SubjectGroup', 'Group'),
                                      ('Guidance', 'Guidance'), ('Smoke', 'Smoke')]:
        if source_col in audit and not audit[source_col].eq(aligned[canonical_col]).all():
            raise ValueError(f'Source {source_col} conflicts with common.load_trials()')
    audit['SubjectID'] = aligned.ParticipantID
    audit['SubjectGroup'] = aligned.Group
    audit['Guidance'] = aligned.Guidance
    audit['Smoke'] = aligned.Smoke
    values = pd.to_numeric(audit[metric], errors='coerce')
    invalid = ~np.isfinite(values) | (values < 0)
    if invalid.any():
        raise ValueError(f'{int(invalid.sum())} invalid {metric} values; cannot silently reduce the 270-trial cohort')
    audit[metric] = values.astype(float)
    audit['Status'] = 'included'
    audit['Reason'] = ''
    # PlotIncluded/IQR flags are display-only and do not change this cohort.
    prepared = audit[['Scenario', 'SubjectID', 'SubjectGroup', metric, 'Guidance', 'Smoke']].copy()
    prepared['SessionCount'] = 1
    prepared['SourceRows'] = audit.SourceRow.astype(str)
    prepared['SourceFiles'] = audit.FileName
    return prepared.sort_values(['Scenario', 'SubjectID']).reset_index(drop=True), audit, []
