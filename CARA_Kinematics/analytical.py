"""User analytical model only. No MuJoCo calls, fitted parameters or CAD queries.
Lengths: mm. Angles: radians. Order: FL_abad, FL_a, FL_b.
Positions: relative to fixed FL_O, world-aligned axes, Z up.
Source: supplied adjusted calculator and typed PDF section 3.
Home offsets were already present in supplied calculator; they are not re-fitted.
"""
import numpy as np
import original_calculator as source


def fk(q):
    a,b,t=source.home_to_equation(*np.asarray(q,float))
    x,y,z=source.fk(a,b,t,tau=1)
    return np.array([x,y,-z])


def ik(target, previous=None):
    x,y,z=np.asarray(target,float)
    candidates=[s for s in source.ik(x,y,-z) if s['tau']==1 and s['eta']==1]
    if not candidates:
        raise ValueError('No IK solution on tau=+1, eta=+1 assembly branch')
    qs=[np.array(source.equation_to_home(*s['angles'])) for s in candidates]
    prior=np.zeros(3) if previous is None else np.asarray(previous,float)
    qs=[prior+(q-prior+np.pi)%(2*np.pi)-np.pi for q in qs]
    return min(qs,key=lambda q:np.linalg.norm(q-prior))


def jacobian(q):
    a,b,t=source.home_to_equation(*np.asarray(q,float))
    g=source.G
    s=(a+b)/2; delta=(a-b)/2
    T2=g.d**2-g.c**2*np.sin(s)**2
    if T2<=1e-12 or abs(np.sin(s))<1e-10:
        raise ValueError('Analytical Jacobian undefined at linkage degeneracy')
    T=np.sqrt(T2); rho=g.c*np.cos(s)+T
    rp=-g.c*np.sin(s)/2-g.c**2*np.sin(s)*np.cos(s)/(2*T)
    sd,cd=np.sin(delta),np.cos(delta)
    JD=np.array([[rp*sd+rho*cd/2,rp*sd-rho*cd/2],
                 [rp*cd-rho*sd/2,rp*cd+rho*sd/2]])
    JC=np.array([[0,-g.c*np.cos(b)],[0,-g.c*np.sin(b)]])
    cp,sp=np.cos(g.phi),np.sin(g.phi)
    R=np.array([[cp,sp],[-sp,cp]])
    JP=(np.eye(2)-g.m/g.d*R)@JD+g.m/g.d*R@JC
    _,y,zdown=source.fk(a,b,t,tau=1)
    J_equation=np.array([[JP[0,0],JP[0,1],0],
                         [np.sin(t)*JP[1,0],np.sin(t)*JP[1,1],zdown],
                         [np.cos(t)*JP[1,0],np.cos(t)*JP[1,1],-y]])
    # q=(t, A_HOME-a, b-B_HOME); flip output Z to the shared world convention.
    dqeq_dq=np.array([[0,-1,0],[0,0,1],[1,0,0]])
    return np.diag([1,1,-1])@J_equation@dqeq_dq
