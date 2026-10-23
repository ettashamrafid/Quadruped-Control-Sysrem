"""Post-run audit: python tests/test_recorded_results.py results/run_1601
Reads saved evidence; does not run MuJoCo or analytical equations.
"""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np


def main():
    root=Path(sys.argv[1]);manifest=json.loads((root/'manifest.json').read_text());n=manifest['config']['samples']
    for name in ['baseline','analytical']:
        folder=root/name;status=json.loads((folder/'status.json').read_text())
        assert status['state']=='complete' and status['samples']==n
        for filename,sha in status['artifacts'].items():
            assert hashlib.sha256((folder/filename).read_bytes()).hexdigest()==sha
        z=np.load(folder/'states.npz',allow_pickle=False)
        with (folder/'samples.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==len(z['time_s'])==n
        for i,row in enumerate(rows):
            assert int(row['sample'])==i
            assert (row['ok']=='True')==bool(z['valid'][i])
            if z['available'][i]:
                pos=np.array([float(row['actual_mm_'+a]) for a in 'xyz'])
                assert np.allclose(pos,z['position_mm'][i],rtol=0,atol=1e-10)
                assert np.isclose(np.linalg.norm(pos-z['target_mm'][i]),float(row['position_error_mm']),rtol=1e-9,atol=1e-10)
        print(name,': saved sample/state consistency passed; valid',int(z['valid'].sum()),'/',n)
    with (root/'comparison/comparison.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    summary=json.loads((root/'comparison/summary.json').read_text());assert len(rows)==n
    masks={'baseline_tracking_error_mm':'baseline_valid','fk_error_mm':'baseline_valid','ik_target_error_mm':'both_valid',
           'two_run_position_error_mm':'both_valid','jacobian_max_entry_error_mm_rad':'jacobian_reference_valid',
           'velocity_error_mm_s':'jacobian_reference_valid','fk_at_ik_measured_error_mm':'analytical_pose_valid'}
    for key,mask in masks.items():
        values=np.array([float(r[key]) for r in rows if r.get(mask)=='True' and r.get(key) not in ('',None)])
        saved=summary['metrics'][key]['valid_only'];assert saved['count']==len(values)
        if len(values):
            assert np.isclose(values.max(),saved['max'],rtol=1e-10,atol=1e-12)
            assert np.isclose(np.sqrt(np.mean(values**2)),saved['rms'],rtol=1e-10,atol=1e-12)
    print('Summary metrics independently recomputed. Audit consistency is not scientific acceptance; inspect acceptance fields.')


if __name__=='__main__':main()
