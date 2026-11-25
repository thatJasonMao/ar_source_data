import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['MPLBACKEND'] = 'Agg'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Results_Spatial_Metric_Surfaces/_timestamp_input'))
spec = importlib.util.spec_from_file_location('scan_candidate', ROOT / 'graph_0911/scan_speed/scan_speed.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PROJECT_ROOT = ROOT
m.EXTRACTOR_DIR = ROOT / 'Results_Spatial_Metric_Surfaces/_timestamp_input'
import numpy as np
import pandas as pd
import extract_timestamp_metrics as ref

# The reference assumes ns timestamps (pandas 2); this host has pandas 3,
# whose parser defaults to us. Normalize only the test adapter, not its core.
original_read_raw = ref.read_raw
def read_raw_ns(*args, **kwargs):
    frame, audit = original_read_raw(*args, **kwargs)
    frame['ParsedTime'] = frame.ParsedTime.astype('datetime64[ns]')
    return frame, audit
ref.read_raw = read_raw_ns

t = np.arange(20) * .02
p = np.zeros((20, 3))
h = np.zeros((20, 3))
g = np.tile([0., 0., 1.], (20, 1))
result = m.summarize_scan_speed(t, p, h, g)
assert result['EyeGazeScanpathRate_deg_s'] == 0
assert result['ValidIntervals'] == 19
h[:, 1] = np.linspace(0, 150, 20)
g = m.Rotation.from_euler('zxy', h[:, [2, 0, 1]], degrees=True).apply(g)
assert m.summarize_scan_speed(t, p, h, g)['EyeGazeScanpathRate_deg_s'] < .0001

# Analytic known answer with a 90-degree turn across unequal time steps:
# centered averaging yields local angles of 0, 26.565, 63.435, 90 degrees.
h[:] = 0
g = np.array([[0., 0., 1.], [0., 0., 1.], [1., 0., 0.], [1., 0., 0.]])
result = m.summarize_scan_speed(np.array([0., .01, .03, .06]), p[:4], h[:4], g)
np.testing.assert_allclose(result['EyeAngle_deg'], 90., atol=1e-10)
np.testing.assert_allclose(result['EyeGazeScanpathRate_deg_s'], 1500., atol=1e-8)

# Invalid middle sample and long gaps cannot create connecting intervals.
g = np.tile([0., 0., 1.], (5, 1))
g[2] = 0
result = m.summarize_scan_speed(np.arange(5)*.02, p[:5], h[:5], g)
assert result['ValidIntervals'] == 2
result = m.summarize_scan_speed(np.array([0., .02, .4, .42, .44]), p[:5], h[:5], g)
assert result['ValidIntervals'] == 2
g[:] = 0
result = m.summarize_scan_speed(np.arange(5)*.02, p[:5], h[:5], g)
assert result['EyeTime_s'] == 0 and np.isnan(result['EyeGazeScanpathRate_deg_s'])
g[:] = [0., 0., 1.]
p_jump = p[:5].copy()
p_jump[2:] = [4., 0., 0.]
assert m.summarize_scan_speed(np.arange(5)*.02, p_jump, h[:5], g)['ValidIntervals'] == 3
print('Synthetic cases passed: static gaze, head compensation, unequal dt, invalid gaze, gaps, no exposure, teleport.')

# Exercise the actual reference process_trial with an all-space classifier.
# This removes only spatial attribution, preserving the original core code.
manifest = pd.read_csv(ROOT / 'graph_0911/basic_indicator/AllSubjects_EvacuationSpeed.csv')
row = manifest.iloc[0]
folder = ROOT / 'RAW_DATA' / row.Scenario / row.Subject
subject_path = folder / row.FileName
eye_path = subject_path.with_name(subject_path.name.replace('_Subject_', '_EyeGaze_'))
pos, _ = ref.read_raw(subject_path, ['TimeStamp', 'PosX', 'PosY', 'PosZ'])
eye, _ = ref.read_raw(eye_path, ['TimeStamp', 'HeadRotX', 'HeadRotY', 'HeadRotZ', 'GazeDirX', 'GazeDirY', 'GazeDirZ'])
ens = eye.ParsedTime.astype('int64').to_numpy()
positions, _ = ref.align_positions(ens, pos.ParsedTime.astype('int64').to_numpy(), pos[['PosX','PosY','PosZ']].to_numpy(float))
trial = SimpleNamespace(Scenario=row.Scenario, SubjectFolder=row.Subject, ParticipantID=row.Subject,
                        FileName=row.FileName, StartTime=eye.TimeStamp.iloc[0], EndTime=eye.TimeStamp.iloc[-1],
                        Guidance=row.Scenario.rsplit('-',1)[0], Smoke=row.Scenario.rsplit('-',1)[1])
classifier = SimpleNamespace(assign=lambda points: np.zeros(len(points), dtype=int),
                             zone_list=[SimpleNamespace(zone_type='Hall', zone_id='AllSpace')])
cells, _, _, _ = ref.process_trial(trial, classifier)
actual = m.summarize_scan_speed((ens-ens[0])/1e9, positions, eye[['HeadRotX','HeadRotY','HeadRotZ']].to_numpy(float), eye[['GazeDirX','GazeDirY','GazeDirZ']].to_numpy(float))
np.testing.assert_allclose(actual['EyeAngle_deg'], cells.EyeAngle_deg.sum(), rtol=1e-12)
np.testing.assert_allclose(actual['EyeTime_s'], cells.EyeTime_s.sum(), rtol=1e-12)
print('Actual reference process_trial parity passed:', row.Scenario, row.Subject, actual)

