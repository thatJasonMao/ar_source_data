"""Canonical A5 trial contract shared by all TIMELINE producers/consumers."""
from pathlib import Path
import csv
import hashlib
import json
import common

KEYS = ('scenario', 'subject', 'src_file')


def canonical_trials(root=None):
    """The only cohort source is common.load_trials(), never a directory scan."""
    if root is not None and Path(root).resolve() != Path(common.RAW_DATA_DIR).resolve():
        raise ValueError('--root must equal common.RAW_DATA_DIR; alternative cohorts are unsupported')
    frame = common.load_trials()
    rows = [dict(scenario=r.Scenario, subject=r.ParticipantID, src_file=r.FileName,
                 group=r.Group, subject_file=str(r.SubjectFile), eye_file=str(r.EyeGazeFile))
            for r in frame.itertuples(index=False)]
    rows.sort(key=trial_key)
    if len(rows) != 270 or len({r['subject'] for r in rows}) != 45:
        raise ValueError('Expected common cohort: 45 participants / 270 trials')
    if len({trial_key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate canonical trial identity')
    return rows


def trial_key(row):
    return tuple(str(row[k]) for k in KEYS)


def cohort_metadata(rows):
    identities = sorted(trial_key(r) for r in rows)
    return dict(cohort_source='common.load_trials()', cohort_trials=len(rows),
                cohort_participants=len({r['subject'] for r in rows}),
                cohort_sha256=hashlib.sha256(json.dumps(identities, ensure_ascii=False,
                    separators=(',', ':')).encode('utf-8')).hexdigest(),
                trial_identity_columns=list(KEYS), trial_id_scheme='canonical_sorted_keys_v1')


def load_timeline(path, root=None):
    """Require the complete exact cohort; normalize ordering by canonical identity.

    Old 292-row timelines are rejected, not silently filtered. Event/frame values
    remain untouched; missing P boundaries are handled by individual analyses.
    """
    canonical = canonical_trials(root)
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not set(KEYS + ('group',)).issubset(reader.fieldnames or []):
            raise ValueError('Timeline lacks trial identity/group columns')
        rows = list(reader)
    indexed = {}
    for row in rows:
        key = trial_key(row)
        if key in indexed:
            raise ValueError(f'Duplicate timeline identity: {key}')
        indexed[key] = row
    expected = {trial_key(r) for r in canonical}
    extra, missing = set(indexed) - expected, expected - set(indexed)
    if extra or missing:
        raise ValueError(f'Timeline differs from common cohort: extra={len(extra)}, '
                         f'missing={len(missing)}; regenerate with Extract_ZoneTimeline.py')
    result = []
    for source in canonical:
        row = indexed[trial_key(source)]
        if row['group'] != source['group']:
            raise ValueError(f'Timeline group differs from common: {trial_key(source)}')
        result.append(dict(row, subject_file=source['subject_file'], eye_file=source['eye_file']))
    return result


def validate_metadata(metadata, rows):
    expected = cohort_metadata(rows)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Legacy or different cohort cache; regenerate canonical timeline products')


def validate_trial_table(table, rows):
    """Every retained point/cell must belong to its canonical key and numeric ID.

    P-level tables may be incomplete after legitimate validity/spatial filtering.
    """
    required = set(KEYS + ('trial_id', 'group'))
    if not required.issubset(table.columns):
        raise ValueError('Trajectory/cell cache lacks canonical identity columns')
    lookup = {trial_key(row): (i, row['group']) for i, row in enumerate(rows, 1)}
    for values in table[list(KEYS) + ['trial_id', 'group']].drop_duplicates().itertuples(index=False, name=None):
        expected = lookup.get(tuple(str(v) for v in values[:3]))
        if expected is None or str(values[4]) != expected[1] or values[3] != expected[0]:
            raise ValueError(f'Trajectory/cell trial identity mismatch: {values}')


def require_fresh_frame_cohort(output, rows):
    """Do not leave legacy, excluded frame exports behind when using --force."""
    output = Path(output)
    if (output / 'frames').exists() and any((output / 'frames').rglob('*.csv.gz')):
        summary = output / 'run_summary.json'
        if not summary.exists():
            raise ValueError('Unversioned frame exports: choose a new --output directory')
        try:
            validate_metadata(json.loads(summary.read_text(encoding='utf-8')), rows)
        except ValueError as exc:
            raise ValueError('Legacy frame exports: choose a new --output directory; '
                             '--force cannot mix old and canonical frame cohorts') from exc
