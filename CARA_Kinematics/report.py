"""Generate plots and a shareable Markdown/PDF report from saved measurements."""
from pathlib import Path
import json
import numpy as np


def make_pdf(path,title,sections,table=None,images=()):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from xml.sax.saxutils import escape
    from PIL import Image as PILImage
    styles=getSampleStyleSheet()
    styles['BodyText'].fontSize=10;styles['BodyText'].leading=14
    story=[Paragraph(escape(title),styles['Title']),Spacer(1,12)]
    for heading,paragraphs in sections:
        story.append(Paragraph(escape(heading),styles['Heading2']))
        for paragraph in paragraphs:
            story.extend([Paragraph(escape(paragraph),styles['BodyText']),Spacer(1,8)])
        if heading=='Numerical summary' and table:
            cells=[[Paragraph(escape(str(x)),styles['BodyText']) for x in row] for row in table]
            tab=Table(cells,colWidths=[210,48,95,95],repeatRows=1)
            tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DDEFEA')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.lightgrey),
                ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            story.extend([tab,Spacer(1,12)])
    for label,path_image in images:
        story.extend([PageBreak(),Paragraph(escape(label),styles['Heading1'])])
        w,h=PILImage.open(path_image).size
        width=min(455,620*w/h)
        story.append(Image(str(path_image),width=width,height=width*h/w))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.drawString(45,25,'CARA | Contacts-off geometric validation')
        canvas.drawRightString(550,25,str(doc.page))
    SimpleDocTemplate(str(path),leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)


def generate(out,c,rows,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(out)/'comparison'
    def series(key):
        return np.array([float(r[key]) if r.get(key) not in ('',None) else np.nan for r in rows])
    def xyz(prefix):return np.column_stack([series(prefix+a) for a in 'xyz'])
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(projection='3d')
    for prefix,label in [('target_mm_','Desired'),('baseline_mm_','CAD baseline'),('analytical_mm_','Analytical IK applied to CAD')]:
        points=xyz(prefix);ax.plot(*points.T,label=label)
    ax.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title='All samples; invalid configurations retained')
    ax.legend();fig.tight_layout();fig.savefig(folder/'paths.png',dpi=180);plt.close(fig)
    keys=['baseline_tracking_error_mm','fk_error_mm','ik_target_error_mm','two_run_position_error_mm',
          'jacobian_max_entry_error_mm_rad','velocity_error_mm_s']
    fig,axes=plt.subplots(6,1,figsize=(11,13),sharex=True)
    for ax,key in zip(axes,keys):
        y=series(key);time=series('nominal_time_s')
        ax.plot(time,y,lw=1)
        flag='jacobian_reference_valid' if key in keys[-2:] else ('baseline_valid' if key in keys[:2] else 'both_valid')
        bad=np.array([not r.get(flag,False) for r in rows]);ax.scatter(time[bad],y[bad],color='red',s=9,label='Invalid reference/pose')
        ax.set_ylabel(key.replace('_',' '),fontsize=8);ax.grid(alpha=.3)
    axes[0].legend();axes[-1].set_xlabel('Nominal trajectory time (s), not relaxation time')
    fig.tight_layout();fig.savefig(folder/'errors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
    for ax,joint in zip(axes,['FL_abad','FL_a','FL_b']):
        ax.plot(series('nominal_time_s'),series('baseline_'+joint+'_deg'),label='Baseline measured')
        ax.plot(series('nominal_time_s'),series('analytical_measured_'+joint+'_deg'),'--',label='Analytical run measured')
        ax.set_ylabel(joint+' (deg)');ax.grid(alpha=.3)
    axes[0].legend();axes[-1].set_xlabel('Nominal trajectory time (s)')
    fig.tight_layout();fig.savefig(folder/'joint_angles.png',dpi=180);plt.close(fig)
    table=[['Metric','Count','Maximum','RMS']]
    for key,d in summary['metrics'].items():
        s=d['valid_only'];fmt=lambda x:'N/A' if x is None else f'{x:.6g}'
        table.append([key.replace('_',' '),s['count'],fmt(s['max']),fmt(s['rms'])])
    manifest=json.loads((Path(out)/'manifest.json').read_text())
    stages={name:json.loads((Path(out)/name/'status.json').read_text()) for name in ['baseline','analytical']}
    checks=summary['acceptance'];passed=all(x['pass'] for x in checks.values())
    sections=[('Experiment and provenance',[
        f"Run signature: {summary['signature']}. Recorded environment: Python {manifest['environment']['python']}, MuJoCo {manifest['environment']['packages']['mujoco']}.",
        f"Contacts OFF; fixed-base leg. {summary['samples']} requested points. Center {c['center_mm']} mm; radius {c['radius_mm']} mm; height {c['height_mm']} mm; one base circle and {c['turns']} rising turns. Nominal duration {c['base_seconds']+c['rise_seconds']} s.",
        'All positions are relative to fixed FL_O with world-aligned axes and Z upward. Angles are ordered FL_abad, FL_a, FL_b. Analytical equations were preserved from the supplied adjusted calculator; retained home-angle mappings are calibration assumptions.']),
        ('Procedure',[
        'Baseline: the moving mocap foot connection is enabled and motor holds are disabled. MuJoCo relaxes the XML mechanism and stores measured angles, XYZ, full qpos/qvel and diagnostic flags.',
        'Analytical run: analytical IK supplies angle targets; the foot connection is disabled and angle holds are enabled. Achieved positions are saved independently. FK is then evaluated at measured baseline angles. The effective CAD Jacobian includes passive joint velocities from differentiated loop closure.',
        'Both simulation stages complete and save files before comparison starts. The comparison does not step the simulation or search for poses.']),
        ('Validity and acceptance',[
        f"Valid reference samples: {summary['baseline_valid']}/{summary['samples']}; valid analytical held poses: {summary['analytical_pose_valid']}/{summary['samples']}; both valid: {summary['both_valid']}/{summary['samples']}.",
        f"Engine thresholds: position {c['position_tolerance_mm']} mm; angle {c['angle_tolerance_deg']} deg; closure {c['closure_tolerance_mm']} mm; speed {c['settled_speed_tolerance_rad_s']} rad/s, with branch and defined-limit checks. Jacobian reference residual must be below 1e-7 m/rad.",
        'IK engine validity does not itself assert a small foot error. Separate comparison thresholds require full coverage: '+json.dumps(checks),
        ('All configured full-coverage comparison checks passed.' if passed else 'At least one comparison check failed or lacks full valid coverage. Inspect failures.csv; no blanket validation verdict is warranted.')]),
        ('Settling diagnostics',[
        'These figures include all finite samples, including failures. Tracking is in mm for baseline and degrees for analytical angle holding.',
        *[name+': maximum closure '+str(group['closure_mm']['max'])+' mm; maximum speed '+str(group['max_joint_speed_rad_s']['max'])+' rad/s; invalid poses '+str(group['invalid_pose_count'])+'; no IK solution '+str(group['no_solution_count']) for name,group in summary['stage_diagnostics'].items()]]),
        ('Numerical summary',[
        'Only eligible valid samples enter the table. summary.json additionally gives all-finite diagnostic statistics, median and 95th percentile. FK/IK errors are in mm, Jacobian entry error in mm/rad, and velocity error in mm/s. RMS is sqrt(mean(error squared)).']),
        ('Interpretation',[
        'FK error compares analytical FK at measured baseline motor angles with baseline foot position. IK error compares the foot achieved using analytical angle commands with the desired target. Two-run error compares the independently achieved positions.',
        f"Velocity uses the same selected joint rates {c['test_joint_rates_deg_s']} deg/s in both Jacobians. It is a local prediction check, not an independent measurement of foot velocity during the videos.",
        'A small geometric error supports agreement with this XML mechanism on this sampled branch and path. It does not prove global correctness, physical accuracy, collision clearance, motor feasibility or real-time tracking. More samples improve path coverage, not physical fidelity.',
        'Physical joint limits were not supplied.' if c['joint_limits_deg'] is None else 'Configured physical limits were supplied; verify their calibration against hardware.']),
        ('Recordings and state files',[
        'Each stage has physics_settling.mp4 when enabled: actual transient states recorded at a fixed physics-step stride, including approach in baseline. Its frame index is not nominal trajectory time. Video is accelerated only by the small stride rounding stated below.',
        f"Frame stride = round(1/(fps*0.001)) steps; fps={c['recording']['fps']}. trajectory_playback.mp4 is nominal-time nearest-saved-pose playback and is labeled as such. It is not a real-time dynamic tracking demonstration.",
        'physics_frames.npz contains the captured transient joint positions and velocities. states.npz contains every final sampled configuration and availability/validity flags. These support inspection but are not bit-exact restart checkpoints of the solver.',
        'Stage recording inventory: '+json.dumps({n:s.get('video_artifacts',{}) for n,s in stages.items()})]),
        ('Files to share',[
        'Share this report, comparison.csv, summary.json, failures.csv, the plots, both stage sample tables and recordings, manifest.json and used_config.json. The full source package and model assets are needed to reproduce the run.',
        'Plots retain invalid points and mark invalid-reference errors in red. Failures must be investigated, not removed. Stage output hashes are checked before analysis; source/config/environment changes require a new run folder.'])]
    text='# CARA contacts-off trajectory results\n\n'
    for title,paragraphs in sections:
        text+='## '+title+'\n\n'+'\n\n'.join(paragraphs)+'\n\n'
        if title=='Numerical summary':
            text+='\n'.join('| '+' | '.join(map(str,r))+' |' for r in [table[0],['---']*4]+table[1:])+'\n\n'
    (folder/'team_results_report.md').write_text(text,encoding='utf-8')
    make_pdf(folder/'team_results_report.pdf','CARA contacts-off trajectory results',sections,table,
             [('Trajectory comparison',folder/'paths.png'),('Errors and invalid samples',folder/'errors.png'),('Measured joint angles',folder/'joint_angles.png')])
