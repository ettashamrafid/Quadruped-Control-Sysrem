"""Independent MuJoCo reference: geometry/constraints from XML, no analytical import.
Positions are solved only with MuJoCo mj_step. Effective Jacobian uses MuJoCo
site Jacobians and the differentiated CAD loop constraint, including passive DOFs.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ACTIVE=['FL_abad','FL_a','FL_b']
PASSIVE=['FL_passive_B','FL_passive_C']

class CadEngine:
    def __init__(self,path,config,contacts=False):
        if contacts: raise ValueError("This workflow requires contacts OFF")
        self.config=config; self.contacts=contacts
        path=Path(path).resolve(); root=ET.parse(path).getroot()
        comp=root.find('compiler'); mesh=Path(comp.get('meshdir','.'))
        comp.set('meshdir',str(mesh if mesh.is_absolute() else path.parent/mesh))
        opt=root.find('option'); opt.set('timestep','.001'); opt.set('solver','Newton')
        opt.set('iterations','200'); opt.set('tolerance','1e-12'); opt.set('gravity','0 0 0')
        flag=opt.find('flag')
        if flag is None: flag=ET.SubElement(opt,'flag')
        flag.set('contact','enable' if contacts else 'disable')
        # CAD meshes use convex hulls. No guessed exclusions for overlapping parts.
        for geom in root.findall('.//geom'):
            mask='1' if contacts and geom.get('type') in ('mesh','plane') else '0'
            geom.set('contype',mask); geom.set('conaffinity',mask)
        limits=config.get('joint_limits_deg')
        if limits is not None:
            for name in ACTIVE:
                lo,hi=limits[name]
                if not np.isfinite([lo,hi]).all() or not lo<hi: raise ValueError('Invalid limits for '+name)
                joint=root.find(".//worldbody//joint[@name='%s']"%name)
                joint.set('limited','true')
                units = [lo, hi] if comp.get('angle', 'degree') == 'degree' else np.deg2rad([lo,hi])
                joint.set('range',' '.join(map(str,units)))
        visual=root.find('visual')
        if visual is None: visual=ET.SubElement(root,'visual')
        glob=visual.find('global')
        if glob is None: glob=ET.SubElement(visual,'global')
        glob.set('offwidth',str(config['recording']['width']))
        glob.set('offheight',str(config['recording']['height']))
        initial=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        d=mujoco.MjData(initial); mujoco.mj_forward(initial,d)
        self.origin=d.site('FL_O').xpos.copy(); self.home=(d.site('FL_P').xpos-self.origin)*1000
        target=ET.SubElement(root.find('worldbody'),'body',name='comparison_target',mocap='true',pos=' '.join(map(str,d.site('FL_P').xpos)))
        ET.SubElement(target,'site',name='comparison_target_site',size='.006',rgba='1 .1 .1 1')
        eq=root.find('equality')
        for e in eq: e.set('solref','.003 1'); e.set('solimp','.9999 .9999 .001')
        ET.SubElement(eq,'connect',name='comparison_foot',site1='FL_P',site2='comparison_target_site',solref='.003 1',solimp='.9999 .9999 .001')
        for name in ACTIVE:
            ET.SubElement(eq,'joint',name='hold_'+name,joint1=name,polycoef='0 0 0 0 0',active='false',solref='.003 1',solimp='.9999 .9999 .001')
        self.xml=ET.tostring(root,encoding='unicode')
        self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.model.dof_damping[:]=.1  # Numerical regularization, not hardware calibration.
        self.data=mujoco.MjData(self.model)
        self.active_q=np.array([self.model.joint(n).qposadr[0] for n in ACTIVE])
        self.active_v=np.array([self.model.joint(n).dofadr[0] for n in ACTIVE])
        self.passive_v=np.array([self.model.joint(n).dofadr[0] for n in PASSIVE])
        self.hold=np.array([self.model.equality('hold_'+n).id for n in ACTIVE])
        self.foot_eq=self.model.equality('comparison_foot').id
        self.target_mocap = int(self.model.body('comparison_target').mocapid[0])
        if self.model.nv != len(ACTIVE) + len(PASSIVE):
            raise ValueError('Expected fixed-base single leg with exactly five hinge DOFs')
        if any(self.model.jnt_type[self.model.joint(n).id] != mujoco.mjtJoint.mjJNT_HINGE for n in ACTIVE + PASSIVE):
            raise ValueError('Only hinge joints supported')
        mujoco.mj_forward(self.model,self.data)
        self.home_assembly_sign=np.sign(self.assembly_measure())

    def assembly_measure(self):
        d=self.data
        B=d.site('FL_B').xpos;C=d.site('FL_C').xpos;D=d.site('FL_D').xpos
        axis=d.xaxis[self.model.joint('FL_a').id]
        return float(np.dot(np.cross(C-B,D-B),axis))

    def position(self): return (self.data.site('FL_P').xpos-self.origin)*1000
    def angles(self): return self.data.qpos[self.active_q].copy()
    def gap(self): return np.linalg.norm(self.data.site('FL_D').xpos-self.data.site('FL_D_from_B').xpos)*1000
    def penetration(self): return max([0.]+[-c.dist*1000 for c in self.data.contact])
    def limit_violation(self):
        m,d=self.model,self.data; result=0.
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                q=d.qpos[m.jnt_qposadr[j]]; lo,hi=m.jnt_range[j]
                result=max(result,lo-q,q-hi)
        return float(np.rad2deg(result))

    def solve(self,kind,value,refresh=lambda:True,on_step=None):
        d,m=self.data,self.model
        d.eq_active[self.foot_eq]=kind=='target'
        d.eq_active[self.hold]=kind=='angles'
        if kind=='target': d.mocap_pos[self.target_mocap]=self.origin+np.asarray(value)/1000
        elif kind=='angles':
            m.eq_data[self.hold,0]=np.asarray(value)-m.qpos0[self.active_q]
        else: raise ValueError('Unknown solver mode')
        cfg=self.config
        for k in range(cfg['max_relaxation_steps']):
            if k%10==0 and not refresh(): return None
            mujoco.mj_step(m,d); mujoco.mj_forward(m,d)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()): raise RuntimeError('Non-finite MuJoCo state')
            if on_step is not None: on_step()
            err=(np.linalg.norm(self.position()-value) if kind=='target' else np.max(np.abs(self.angles()-value))*180/np.pi)
            threshold=cfg['position_tolerance_mm'] if kind=='target' else cfg['angle_tolerance_deg']
            if k+1>=cfg['min_relaxation_steps'] and err<threshold*.1 and self.gap()<cfg['closure_tolerance_mm']*.1 and np.max(np.abs(d.qvel))<1e-4: break
        tracking=(float(np.linalg.norm(self.position()-value)) if kind=='target' else float(np.rad2deg(np.max(np.abs(self.angles()-value)))))
        assembly=self.assembly_measure()
        branch_ok=bool(abs(assembly)>1e-10 and np.sign(assembly)==self.home_assembly_sign)
        speed=float(np.max(np.abs(d.qvel)))
        settled=bool(speed<cfg['settled_speed_tolerance_rad_s'])
        ok=(settled and branch_ok and tracking< (cfg['position_tolerance_mm'] if kind=='target' else cfg['angle_tolerance_deg']) and self.gap()<cfg['closure_tolerance_mm'] and self.penetration()<cfg['penetration_tolerance_mm'] and self.limit_violation()<cfg['angle_tolerance_deg'])
        return dict(ok=bool(ok),settled=settled,max_joint_speed_rad_s=speed,home_assembly_branch=branch_ok,assembly_measure_m2=assembly,steps=k+1,tracking=tracking,closure_mm=float(self.gap()),contacts=int(d.ncon),penetration_mm=float(self.penetration()),limit_violation_deg=self.limit_violation())

    def site_jac(self,name):
        j=np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.model,self.data,j,None,self.model.site(name).id)
        return j

    def jacobian(self):
        A=self.site_jac('FL_D_from_B')-self.site_jac('FL_D')
        Ap=A[:,self.passive_v]; Aa=A[:,self.active_v]
        B,_,rank,_=np.linalg.lstsq(Ap,-Aa,rcond=None)
        if rank<2: raise ValueError('CAD loop velocity constraint is singular')
        residual=float(np.max(np.abs(Ap@B+Aa)))
        J=self.site_jac('FL_P')
        return (J[:,self.active_v]+J[:,self.passive_v]@B)*1000,residual,float(np.linalg.cond(Ap))
