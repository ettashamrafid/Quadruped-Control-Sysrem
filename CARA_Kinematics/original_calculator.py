#!/usr/bin/env python3
"""Batch analytical CARA FK/IK; no menus or MuJoCo required.
Public inputs/outputs: CAD-home joint angles in degrees; positions in mm,
relative to FL_O with world-aligned axes (Z upward).
The original derivation functions fk()/ik() retain their Z-down convention.
Uses the supplied geometry and home-angle mapping. No joint limits or
collision model were supplied: results are mathematical configurations.
"""
import math as M
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Geometry:
    c: float = 100.0
    d: float = 150.0
    m: float = 45.0
    phi: float = M.radians(150.0)
    e: float = 47.125
    h: float = 43.2319326424363
    @property
    def n(self):
        return M.sqrt(self.d**2 + self.m**2 - 2*self.d*self.m*M.cos(self.phi))

G = Geometry()
A_HOME = M.atan2(77.192140524951, 63.5717975298537)
B_HOME = M.atan2(77.6423800479765, 63.0211140847763)
REFERENCE_HEIGHT_MM = 400.0  # Display reference; does not affect local FK.

def home_to_equation(q_abad, q_a, q_b):
    return A_HOME-q_a, B_HOME+q_b, q_abad

def equation_to_home(a, b, t):
    return wrap(t), wrap(A_HOME-a), wrap(b-B_HOME)

POSITION_TOL = 1e-6  # mm, mathematical verification tolerance

def wrap(a):
    return (a + M.pi) % (2*M.pi) - M.pi

def root(value, scale=1.0):
    # Only tolerate floating-point roundoff, not unreachable geometry.
    if value < -1e-12 * max(1.0, scale):
        raise ValueError('Target/configuration has no real geometric solution.')
    return M.sqrt(max(0.0, value))

def foot_from_dc(D, C, g=G):
    dx, dl = C[0]-D[0], C[1]-D[1]
    k, cp, sp = g.m/g.d, M.cos(g.phi), M.sin(g.phi)
    return (D[0]+k*(dx*cp+dl*sp), D[1]+k*(-dx*sp+dl*cp))

def fk(theta_a, theta_b, t, tau=1, g=G):
    """PDF Eqs. 2.1-2.15; return local (x,y,z) in mm."""
    if not all(M.isfinite(v) for v in (theta_a, theta_b, t)):
        raise ValueError('Angles must be finite.')
    if tau not in (-1, 1):
        raise ValueError('FK branch tau must be +1 or -1.')
    s, delta = (theta_a+theta_b)/2, (theta_a-theta_b)/2
    if abs(M.sin(s)) < 1e-10:
        raise ValueError('Non-unique FK: B=C. The two distal circles coincide.')
    T = root(g.d*g.d-g.c*g.c*M.sin(s)**2, g.d*g.d)
    rho = g.c*M.cos(s)+tau*T
    D = (rho*M.sin(delta), rho*M.cos(delta))
    C = (-g.c*M.sin(theta_b), g.c*M.cos(theta_b))
    x, ell = foot_from_dc(D, C, g)
    L = g.h+ell
    return (x, g.e*M.cos(t)+L*M.sin(t), -g.e*M.sin(t)+L*M.cos(t))

def ik(x, y, z, g=G):
    """PDF Eqs. 1.5-1.28, signed foot relation, then FK validation.

    Return dictionaries containing angles (a,b,t), tau and residual_mm.
    Both radial signs and all discrete planar candidates are considered.
    """
    if not all(M.isfinite(v) for v in (x,y,z)):
        raise ValueError('Coordinates must be finite.')
    radial = root(y*y+z*z-g.e*g.e, y*y+z*z+g.e*g.e)
    if y*y+z*z < 1e-20:
        raise ValueError('Abduction angle is not determined at this target.')
    n = g.n
    if n < 1e-10:
        raise ValueError('Degenerate rigid foot triangle: CP=0.')
    u = (n*n+g.d*g.d-g.m*g.m)/(2*n)
    v = root(g.d*g.d-u*u, g.d*g.d)
    solutions = []
    for eta in (1,-1):
        L = eta*radial
        ell = L-g.h
        t = M.atan2(L*y-g.e*z, g.e*y+L*z)
        f2 = x*x+ell*ell
        if f2 < 1e-20:
            continue
        k = (f2+g.c*g.c-n*n)/(2*g.c)
        try:
            kb_abs = root(f2-k*k, f2+k*k)
        except ValueError:
            continue
        for eb in (1,-1):
            kb = eb*kb_abs
            b = M.atan2(ell*kb-x*k, ell*k+x*kb)
            C = (-g.c*M.sin(b), g.c*M.cos(b))
            dx, dl = x-C[0], ell-C[1]
            for sigma in (1,-1):
                D = (C[0]+u/n*dx+sigma*v/n*dl,
                     C[1]+u/n*dl-sigma*v/n*dx)
                if M.dist(foot_from_dc(D,C,g),(x,ell)) > POSITION_TOL:
                    continue
                r2 = D[0]**2+D[1]**2
                if r2 < 1e-20:
                    continue
                ka = (r2+g.c*g.c-g.d*g.d)/(2*g.c)
                try:
                    ka_abs = root(r2-ka*ka,r2+ka*ka)
                except ValueError:
                    continue
                for ea in (1,-1):
                    kap = ea*ka_abs
                    a = M.atan2(D[0]*ka+D[1]*kap, D[1]*ka-D[0]*kap)
                    for tau in (1,-1):
                        try:
                            error = M.dist(fk(a,b,t,tau,g),(x,y,z))
                        except ValueError:
                            continue
                        if error > POSITION_TOL:
                            continue
                        angles = (wrap(a),wrap(b),wrap(t))
                        if any(max(abs(wrap(i-j)) for i,j in zip(angles,s['angles'])) < 1e-8
                               and tau == s['tau'] for s in solutions):
                            continue
                        solutions.append(dict(angles=angles,tau=tau,residual_mm=error,eta=eta))
    return solutions

# Angles in degrees: (abduction, motor_a, motor_b).
POSES = [
    (0, 0, 0),
    (30, 30, 30),
    (0, 30, 30),
    (0, 45, 30),
    (20, 0, 70),
    (-10, 30, 60),
    (-30, 60, 15),
]

# Foot coordinates relative to FL_O, with world-aligned axes.
# Units: millimetres.
TARGETS_MM = np.array([
    [-175.066750, -86.634908, -244.306062],
    [-141.263017,  13.122330, -196.961627],
    [ -86.856931, 106.574904, -155.027860],
    [-155.420190,  47.125000, -240.661209],
    [-118.384034,  47.125000, -248.279094],
    [-118.384034, 164.950994, -191.453502],
    [   0.000000,  47.125000, -280.000000],
])


def calculate_fk(poses=POSES):
    """Return an N-by-3 array of (x,y,z) in mm, Z upward.

    Use the home assembly branch tau=+1. Invalid rows raise ValueError.
    """
    rows = []
    for index, pose in enumerate(poses, 1):
        if len(pose) != 3:
            raise ValueError(f"FK row {index}: expected three angles.")
        a, b, t = home_to_equation(*map(M.radians, pose))
        x, y, z_down = fk(a, b, t, tau=1)
        rows.append((x, y, -z_down))
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def calculate_ik(targets_mm=TARGETS_MM):
    """Return N-by-3 degrees: (FL_abad, FL_a, FL_b).

    Restrict to the home assembly (tau=+1, positive downward radial L).
    Among verified candidates, choose minimum squared wrapped joint angles
    from CAD home. Targets are solved independently, not as a trajectory.
    No solution on this branch raises ValueError rather than returning angles
    for a different assembly. Joint limits/collisions are not checked.
    """
    rows = []
    for index, target in enumerate(targets_mm, 1):
        if len(target) != 3:
            raise ValueError(f"IK row {index}: expected three coordinates.")
        x, y, z = map(float, target)
        candidates = [s for s in ik(x, y, -z)
                      if s['tau'] == 1 and s['eta'] == 1]
        if not candidates:
            raise ValueError(f"IK row {index}: no solution on the home assembly branch.")
        chosen = min(candidates, key=lambda s: sum(
            q*q for q in equation_to_home(*s['angles'])))
        rows.append(tuple(map(M.degrees, equation_to_home(*chosen['angles']))))
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def print_rows(header, rows):
    print(header)
    for row in rows:
        print(" ".join(f"{v:14.6f}" for v in row))


def main():
    print_rows("FK: x_mm y_mm z_mm", calculate_fk())
    print()
    print_rows("IK (degrees): FL_abad FL_a FL_b", calculate_ik())


if __name__ == '__main__':
    main()
