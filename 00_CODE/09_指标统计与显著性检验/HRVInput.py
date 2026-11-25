"""Shared, audited HRV three-stage input for plots and repeated-measures tests."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

MODES = {'A': ['StaticSign', 'ArrowRibbon', 'Full'],
         'B': ['StaticSign', 'MiniMap', 'Full'],
         'C': ['StaticSign', 'TextBar', 'Full']}
LABELS = {'StaticSign': 'Traditional', 'ArrowRibbon': 'Arrow', 'MiniMap': 'Map',
          'TextBar': 'Text', 'Full': 'Full'}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_hrv(root, mapping_path=None):
    root = Path(root).resolve()
    folder = root/'INDICATORS'/'questionnaire_and_hrv'
    source = folder/'HRV_2.xlsx'
    mapping_path = Path(mapping_path) if mapping_path else folder/'HRV_participant_mapping.csv'
    common = load_module('hrv_common', root/'common.py')
    cohort = common.load_trials().drop_duplicates('ParticipantID').set_index('ParticipantID')
    mapping = pd.read_csv(mapping_path, keep_default_na=False)
    for col in ['RawParticipantID', 'ParticipantID']:
        if mapping[col].eq('').any() or mapping[col].duplicated().any():
            raise ValueError(f'Missing/duplicate mapping {col}')
    sheet = pd.read_excel(source, sheet_name='Sheet1', header=None)
    if sheet.shape != (26, 46):
        raise ValueError('HRV_2 sheet structure changed; review before parsing')
    raw_ids = sheet.iloc[1, 1:].tolist()
    if len(set(raw_ids)) != 45 or set(raw_ids) != set(mapping.RawParticipantID):
        raise ValueError('Source/mapping participant set mismatch')
    lookup = mapping.set_index('RawParticipantID')
    rows, audit = [], []
    for group, header in [('A', 10), ('B', 16), ('C', 22)]:
        expected_labels = ['传统', {'A':'箭头','B':'地图','C':'文本'}[group], '全量']
        if sheet.iloc[header+1:header+4, 0].tolist() != expected_labels:
            raise ValueError('Stage labels/order changed')
        for col in range(1, 16):
            raw_id = sheet.iloc[header, col]
            mapped = lookup.loc[raw_id]
            pid = mapped.ParticipantID
            if mapped.Group != group or raw_id.rsplit('_', 1)[1][0] != group:
                raise ValueError('Mapping/source group mismatch')
            included = pid in cohort.index
            if included and cohort.loc[pid, 'Group'] != group:
                raise ValueError('Mapping/common group mismatch')
            if mapped.CohortStatus != ('included' if included else 'excluded_by_common'):
                raise ValueError('Archived inclusion status differs from current common')
            top_col = raw_ids.index(raw_id)+1
            for stage, mode in enumerate(MODES[group], 1):
                value = float(sheet.iloc[header+stage, col])
                pair = sheet.iloc[2+(stage-1)*2:4+(stage-1)*2, top_col].to_numpy(dtype=float)
                if not np.isfinite(value) or not np.isfinite(pair).all():
                    raise ValueError(f'Missing/nonfinite HRV value: {raw_id}/{mode}')
                reference = float(pair.mean())
                # All three source blocks occupy columns B:P; retain exact Excel cells.
                cell = f'{chr(65+col)}{header+stage+1}'
                record = dict(ParticipantID=pid, RawParticipantID=raw_id, Group=group,
                              StageOrder=stage, Guidance=mode, HRVChange=value,
                              MappingMethod=mapped.MappingMethod, SourceFile='HRV_2.xlsx',
                              SourceSheet='Sheet1', SourceCell=cell)
                audit.append(dict(**record, Included=included, SixRecordPairMean=reference,
                                  DifferenceFromPairMean=value-reference,
                                  MatchesPairMean=bool(np.isclose(value,reference,rtol=0,atol=1e-8))))
                if included:
                    rows.append(record)
    data = pd.DataFrame(rows).sort_values(['Group','ParticipantID','StageOrder']).reset_index(drop=True)
    if data.duplicated(['ParticipantID','StageOrder']).any():
        raise ValueError('Duplicate participant-stage records')
    for pid, part in data.groupby('ParticipantID'):
        if part.StageOrder.tolist() != [1,2,3]:
            raise ValueError(f'Incomplete stage set: {pid}')
    coverage = cohort[['Group']].reset_index()
    coverage['Status'] = np.where(coverage.ParticipantID.isin(data.ParticipantID), 'included', 'missing_from_HRV_2')
    coverage = coverage.sort_values(['Group','ParticipantID'])
    return data, pd.DataFrame(audit), coverage, source, mapping_path


def group_wide(data, group):
    part = data[data.Group.eq(group)]
    wide = part.pivot(index='ParticipantID', columns='Guidance', values='HRVChange').reindex(columns=MODES[group]).sort_index()
    if len(wide)<3 or wide.isna().any().any():
        raise ValueError(f'Insufficient/incomplete repeated measures in group {group}')
    return wide
