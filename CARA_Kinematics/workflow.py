"""Contacts-off baseline -> analytical run -> comparison. Nothing runs on import."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import traceback
import numpy as np
import mujoco
from cad_engine import CadEngine, ACTIVE

HERE = Path(__file__).resolve().parent


def trajectory(c):
    n = c['samples']
    if not isinstance(n,int) or n < 5:
        raise ValueError('samples must be an integer >= 5')
    for key in ('base_seconds','rise_seconds','radius_mm','height_mm'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Invalid '+key)
    if not np.isfinite(c['center_mm']).all() or len(c['center_mm']) != 3:
        raise ValueError('center_mm must contain three finite coordinates')
    t = np.linspace(0,c['base_seconds']+c['rise_seconds'],n)
    smooth = lambda u: 10*u**3-15*u**4+6*u**5
    v = smooth(np.clip((t-c['base_seconds'])/c['rise_seconds'],0,1))
    theta = np.where(t <= c['base_seconds'],2*np.pi*smooth(np.clip(t/c['base_seconds'],0,1)),
                     2*np.pi+2*np.pi*c['turns']*v)
    radius = c['radius_mm']*(1-v)
    return t, np.asarray(c['center_mm'])+np.column_stack((radius*np.cos(theta),radius*np.sin(theta),c['height_mm']*v))


def write_json(path,obj):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    return {p.relative_to(HERE).as_posix():digest(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and (p.parent==HERE and p.suffix in ('.py','.xml') or p.parent==HERE/'meshes')}


def environment():
    return {'python':platform.python_version(),'platform':platform.platform(),
            'packages':{n:importlib.metadata.version(n) for n in
                        ['mujoco','numpy','matplotlib','imageio','imageio-ffmpeg','Pillow','reportlab']}}


def signature(c):
    return hashlib.sha256(json.dumps({'config':c,'sources':source_manifest(),'environment':environment()},sort_keys=True).encode()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save_csv(path,rows):
    if not rows:
        return
    path = Path(path)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def vec(prefix,value):
    return {prefix+a:float(v) for a,v in zip('xyz',value)}


def angle_columns(prefix,q):
    return {prefix+n+'_deg':float(x) for n,x in zip(ACTIVE,np.rad2deg(q))}


def checkpoint(folder,rows,states,velocities,positions,valid,available,t,p,sig):
    save_csv(folder/'samples.csv',rows)
    tmp=folder/'states.tmp.npz'
    count=len(rows)
    np.savez_compressed(tmp,qpos=np.asarray(states),qvel=np.asarray(velocities),position_mm=np.asarray(positions),
                        valid=np.asarray(valid,bool),available=np.asarray(available,bool),
                        time_s=t[:count],target_mm=p[:count],signature=sig)
    tmp.replace(folder/'states.npz')


def verify_stage(folder,sig,expected):
    status=json.loads((folder/'status.json').read_text())
    if status.get('state')!='complete' or status.get('samples')!=expected or status.get('signature')!=sig:
        raise ValueError('Incomplete or incompatible stage: '+str(folder))
    for name,sha in status['artifacts'].items():
        if digest(folder/name)!=sha:
            raise ValueError('Stage data changed: '+str(folder/name))
    z=np.load(folder/'states.npz',allow_pickle=False)
    if len(z['time_s'])!=expected or str(z['signature'].item())!=sig:
        raise ValueError('State count/signature mismatch')
    return z


def run_stage(kind,args,c,t,p,sig):
    folder=args.out/kind
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Use a new output directory; will not overwrite '+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
    status={'state':'running','signature':sig,'samples':0,'kind':kind,'contacts':'off'}
    write_json(folder/'status.json',status)
    e=CadEngine(HERE/'cara_one_leg.xml',c,False)
    rows=[];states=[];velocities=[];positions=[];valid=[];available=[]
    cap=None
    previous=np.zeros(3)
    try:
        from recording import Capture
        cap=Capture(e,p,c,folder,kind.upper(),preview=args.preview)
        if kind=='baseline':
            for target in np.linspace(e.home,p[0],c['approach_samples']):
                cap.set_sample(-1,target)
                if e.solve('target',target,cap.refresh,cap.on_step) is None:
                    raise InterruptedError('Viewer closed')
        else:
            # Load equations only in the analytical stage, never in baseline.
            import analytical
        for i,target in enumerate(p):
            cap.set_sample(i,target)
            row={'sample':i,'nominal_time_s':float(t[i]),**vec('target_mm_',target)}
            solved=True
            if kind=='analytical':
                e.data.mocap_pos[e.target_mocap]=e.origin+target/1000
                try:
                    requested=analytical.ik(target,previous)
                    if np.asarray(requested).shape!=(3,) or not np.isfinite(requested).all():
                        raise ValueError('IK returned invalid angles')
                    row.update(angle_columns('requested_',requested))
                    row['ik_fk_self_error_mm']=float(np.linalg.norm(analytical.fk(requested)-target))
                except (ValueError,np.linalg.LinAlgError) as ex:
                    solved=False
                    row.update(error=str(ex),status='IK_NO_SOLUTION',ok=False)
            if solved:
                d=e.solve('target' if kind=='baseline' else 'angles',target if kind=='baseline' else requested,
                          cap.refresh,cap.on_step)
                if d is None:
                    raise InterruptedError('Viewer closed')
                row.update(d)
                row.update(angle_columns('measured_',e.angles()))
                position=e.position().copy()
                row.update(vec('actual_mm_',position))
                row['position_error_mm']=float(np.linalg.norm(position-target))
                row['simulation_time_s']=float(e.data.time)
                row['status']='VALID' if d['ok'] else 'INVALID_POSE'
                state=e.data.qpos.copy();velocity=e.data.qvel.copy()
                if kind=='analytical' and d['ok']:
                    previous=e.angles().copy()
                cap.trail.append(position.copy())
            else:
                position=np.full(3,np.nan);state=np.full(e.model.nq,np.nan);velocity=np.full(e.model.nv,np.nan)
            rows.append(row);states.append(state);velocities.append(velocity);positions.append(position)
            valid.append(bool(row['ok']));available.append(solved)
            if i%25==0 or i==len(p)-1:
                checkpoint(folder,rows,states,velocities,positions,valid,available,t,p,sig)
                status['samples']=len(rows);write_json(folder/'status.json',status)
                print(f'{kind}: {i+1}/{len(p)} | {row["status"]}',flush=True)
            if not cap.refresh():
                raise InterruptedError('Viewer closed')
        cap.trajectory(states,t,p,positions,valid,available)
        cap.close();cap=None
        status.update(state='complete',samples=len(rows),valid_samples=sum(valid),
                      artifacts={name:digest(folder/name) for name in ['samples.csv','states.npz']},
                      video_artifacts={x.name:digest(x) for x in folder.glob('*.mp4')})
        write_json(folder/'status.json',status)
    except BaseException as ex:
        if rows:
            checkpoint(folder,rows,states,velocities,positions,valid,available,t,p,sig)
        status.update(state='incomplete',samples=len(rows),error=str(ex))
        write_json(folder/'status.json',status)
        raise
    finally:
        if cap is not None:
            cap.close()


def stats(values,mask):
    a=np.asarray(values,float)[np.asarray(mask,bool)]
    a=a[np.isfinite(a)]
    return {'count':len(a),'max':float(a.max()) if len(a) else None,
            'rms':float(np.sqrt(np.mean(a*a))) if len(a) else None,
            'median':float(np.median(a)) if len(a) else None,
            'p95':float(np.percentile(a,95)) if len(a) else None}


def compare(args,c,t,p,sig):
    """Only reads saved stage outputs; no mj_step and no new poses solved."""
    b=verify_stage(args.out/'baseline',sig,len(t));a=verify_stage(args.out/'analytical',sig,len(t))
    if not (np.array_equal(b['time_s'],a['time_s']) and np.array_equal(b['target_mm'],a['target_mm']) and np.array_equal(b['target_mm'],p)):
        raise ValueError('Stage target/time arrays differ')
    import analytical
    e=CadEngine(HERE/'cara_one_leg.xml',c,False)
    rows=[]
    qd=np.deg2rad(c['test_joint_rates_deg_s'])
    for i in range(len(t)):
        row={'sample':i,'nominal_time_s':float(t[i]),**vec('target_mm_',p[i]),
             'baseline_valid':bool(b['valid'][i]),'analytical_pose_valid':bool(a['valid'][i]),
             'both_valid':bool(b['valid'][i] and a['valid'][i])}
        row.update(vec('baseline_mm_',b['position_mm'][i]));row.update(vec('analytical_mm_',a['position_mm'][i]))
        row['baseline_tracking_error_mm']=float(np.linalg.norm(b['position_mm'][i]-p[i]))
        row['ik_target_error_mm']=float(np.linalg.norm(a['position_mm'][i]-p[i]))
        row['two_run_position_error_mm']=float(np.linalg.norm(a['position_mm'][i]-b['position_mm'][i]))
        row.update(vec('baseline_minus_target_mm_',b['position_mm'][i]-p[i]))
        row.update(vec('analytical_minus_target_mm_',a['position_mm'][i]-p[i]))
        row.update(vec('analytical_minus_baseline_mm_',a['position_mm'][i]-b['position_mm'][i]))
        e.data.qpos[:]=b['qpos'][i];e.data.qvel[:]=0;mujoco.mj_forward(e.model,e.data)
        q=e.angles();row.update(angle_columns('baseline_',q))
        try:
            pf=analytical.fk(q);row.update(vec('fk_mm_',pf))
            row['fk_error_mm']=float(np.linalg.norm(pf-b['position_mm'][i]))
            row.update(vec('fk_minus_baseline_mm_',pf-b['position_mm'][i]))
        except (ValueError,np.linalg.LinAlgError) as ex:
            row['fk_failure']=str(ex)
        try:
            ja=analytical.jacobian(q);jc,res,condition=e.jacobian()
            row['jacobian_max_entry_error_mm_rad']=float(np.max(np.abs(ja-jc)))
            row['closure_velocity_residual_m_rad']=float(res)
            row['passive_condition']=float(condition)
            row['analytical_condition']=float(np.linalg.cond(ja))
            row['jacobian_reference_valid']=bool(b['valid'][i] and res<1e-7)
            va=ja@qd;vc=jc@qd
            row.update(vec('velocity_analytical_mm_s_',va));row.update(vec('velocity_cad_mm_s_',vc))
            row['velocity_error_mm_s']=float(np.linalg.norm(va-vc))
            for r in range(3):
                for j in range(3):
                    row[f'J_analytical_{r}{j}_mm_rad']=float(ja[r,j]);row[f'J_cad_{r}{j}_mm_rad']=float(jc[r,j])
        except (ValueError,np.linalg.LinAlgError) as ex:
            row['jacobian_failure']=str(ex);row['jacobian_reference_valid']=False
        if a['available'][i]:
            qa=a['qpos'][i,e.active_q];row.update(angle_columns('analytical_measured_',qa))
            # Wrapped difference is descriptive: multiple IK branches can be valid.
            dq=(qa-q+np.pi)%(2*np.pi)-np.pi
            row.update(angle_columns('angle_difference_',dq))
            try:
                row['fk_at_ik_measured_error_mm']=float(np.linalg.norm(analytical.fk(qa)-a['position_mm'][i]))
            except ValueError as ex:
                row['fk_at_ik_failure']=str(ex)
        rows.append(row)
    folder=args.out/'comparison'
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Comparison exists; preserve it or choose a new output directory')
    folder.mkdir(exist_ok=True)
    # Replace NaN/inf with blank CSV cells, preserving explicit validity fields.
    for row in rows:
        for key,v in list(row.items()):
            if isinstance(v,float) and not np.isfinite(v):row[key]=''
    save_csv(folder/'comparison.csv',rows)
    specs={'baseline_tracking_error_mm':'baseline_valid','fk_error_mm':'baseline_valid',
           'ik_target_error_mm':'both_valid','two_run_position_error_mm':'both_valid',
           'jacobian_max_entry_error_mm_rad':'jacobian_reference_valid','velocity_error_mm_s':'jacobian_reference_valid',
           'fk_at_ik_measured_error_mm':'analytical_pose_valid'}
    summary={'signature':sig,'samples':len(t),'contacts':'off','baseline_valid':int(b['valid'].sum()),
             'analytical_pose_valid':int(a['valid'].sum()),'both_valid':sum(r['both_valid'] for r in rows),
             'metrics':{},'acceptance':{},'rates_deg_s':c['test_joint_rates_deg_s'],
             'interpretation':'Settled-pose geometric agreement with the supplied XML; not physical accuracy or real-time tracking.'}
    summary['stage_diagnostics']={}
    for name in ['baseline','analytical']:
        stage_rows=read_csv(args.out/name/'samples.csv')
        group={}
        for field in ['closure_mm','max_joint_speed_rad_s','tracking','steps','position_error_mm','ik_fk_self_error_mm']:
            values=[float(r[field]) if r.get(field) not in ('',None) else np.nan for r in stage_rows]
            group[field]=stats(values,[True]*len(values))
        group['tracking_units']='mm' if name=='baseline' else 'degree'
        group['invalid_pose_count']=sum(r['status']=='INVALID_POSE' for r in stage_rows)
        group['no_solution_count']=sum(r['status']=='IK_NO_SOLUTION' for r in stage_rows)
        summary['stage_diagnostics'][name]=group
    for key,mask_key in specs.items():
        values=[r.get(key) if r.get(key) not in ('',None) else np.nan for r in rows]
        mask=[bool(r.get(mask_key,False)) for r in rows]
        summary['metrics'][key]={'valid_only':stats(values,mask),'all_finite_diagnostic_only':stats(values,[True]*len(rows))}
    for key,tol in [('fk_error_mm',c['comparison_position_tolerance_mm']),('ik_target_error_mm',c['comparison_position_tolerance_mm']),
                    ('jacobian_max_entry_error_mm_rad',c['jacobian_tolerance_mm_rad']),('velocity_error_mm_s',c['velocity_tolerance_mm_s'])]:
        s=summary['metrics'][key]['valid_only']
        summary['acceptance'][key]={'tolerance':tol,'full_coverage':s['count']==len(t),
                                    'pass':bool(s['count']==len(t) and s['max'] is not None and s['max']<=tol)}
    write_json(folder/'summary.json',summary)
    failures=[r for r in rows if not r['both_valid'] or any(r.get(k,'')=='' or float(r[k])>c['comparison_position_tolerance_mm'] for k in ['fk_error_mm','ik_target_error_mm']) or not r.get('jacobian_reference_valid',False)]
    # Retain failures even if the primary position metrics are small.
    failure_ids={int(r['sample']) for r in failures}
    for r in rows:
        if any(r.get(key,'')=='' or float(r[key])>tol for key,tol in [('jacobian_max_entry_error_mm_rad',c['jacobian_tolerance_mm_rad']),('velocity_error_mm_s',c['velocity_tolerance_mm_s'])]):
            failure_ids.add(r['sample'])
    save_csv(folder/'failures.csv',[r for r in rows if r['sample'] in failure_ids])
    if not failure_ids:
        (folder/'failures.csv').write_text('sample,reason\n',encoding='utf-8')
    from report import generate
    generate(args.out,c,rows,summary)
    write_json(folder/'status.json',{'state':'complete','signature':sig,'samples':len(rows),
                                  'artifacts':{x.name:digest(x) for x in folder.iterdir() if x.is_file() and x.name!='status.json'}})
    print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['baseline','analytical','compare','all'],default='all')
    ap.add_argument('--config',type=Path,default=HERE/'config.json')
    ap.add_argument('--out',type=Path,required=True,help='One new run folder; reuse only for successive stages')
    ap.add_argument('--preview',action='store_true',help='Show live viewer as well as saving recordings')
    args=ap.parse_args();args.out=args.out.resolve();c=json.loads(args.config.read_text())
    for key in ['approach_samples','max_relaxation_steps','min_relaxation_steps']:
        if not isinstance(c[key],int) or c[key]<1:raise ValueError('Positive integer required: '+key)
    if c['min_relaxation_steps']>c['max_relaxation_steps']:raise ValueError('min relaxation exceeds max')
    for key in ['position_tolerance_mm','closure_tolerance_mm','angle_tolerance_deg','settled_speed_tolerance_rad_s',
                'comparison_position_tolerance_mm','jacobian_tolerance_mm_rad','velocity_tolerance_mm_s']:
        if not np.isfinite(c[key]) or c[key]<=0:raise ValueError('Positive finite tolerance required: '+key)
    rc=c['recording']
    if any(not isinstance(rc[k],int) or rc[k]<=0 for k in ['fps','width','height']):raise ValueError('Invalid video dimensions/fps')
    if rc['width']%2 or rc['height']%2:raise ValueError('Video dimensions must be even')
    if not np.isfinite(c['turns']) or c['turns']<0:raise ValueError('turns must be finite and nonnegative')
    if np.asarray(c['test_joint_rates_deg_s']).shape!=(3,) or not np.isfinite(c['test_joint_rates_deg_s']).all():raise ValueError('Three finite joint rates required')
    t,p=trajectory(c);sig=signature(c);args.out.mkdir(parents=True,exist_ok=True)
    manifest_path=args.out/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['signature']!=sig:raise ValueError('Sources/config/environment changed; start a new run folder')
    else:
        write_json(manifest_path,{'signature':sig,'config':c,'sources':source_manifest(),'environment':environment(),
                                 'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                                 'contacts':'off','scope':'fixed-base leg; independent angles FL_abad, FL_a, FL_b'})
        write_json(args.out/'used_config.json',c)
        save_csv(args.out/'trajectory.csv',[{'sample':i,'nominal_time_s':float(ti),**vec('target_mm_',pi)} for i,(ti,pi) in enumerate(zip(t,p))])
    if args.stage in ('baseline','all'):
        run_stage('baseline',args,c,t,p,sig)
    if args.stage in ('analytical','all'):
        verify_stage(args.out/'baseline',sig,len(t))
        run_stage('analytical',args,c,t,p,sig)
    if args.stage in ('compare','all'):
        compare(args,c,t,p,sig)


if __name__=='__main__':
    try:
        main()
    except (Exception,KeyboardInterrupt):
        traceback.print_exc()
        sys.exit(1)
