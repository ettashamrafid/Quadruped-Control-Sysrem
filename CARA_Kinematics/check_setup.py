"""Explicit preflight: no physics steps. Run locally before a new experiment."""
from pathlib import Path
import json
import numpy as np
import mujoco
import imageio_ffmpeg
from workflow import trajectory, environment
from cad_engine import CadEngine


def main():
    here=Path(__file__).resolve().parent
    c=json.loads((here/'config.json').read_text());t,p=trajectory(c)
    e=CadEngine(here/'cara_one_leg.xml',c,False)
    assert e.data.ncon==0
    assert np.isfinite(e.position()).all()
    assert e.model.nq==5 and e.model.nv==5
    j,res,_=e.jacobian();assert j.shape==(3,3) and np.isfinite(j).all()
    print(json.dumps(environment(),indent=2))
    print('Frames:',c['recording'],'; samples:',len(t))
    print('FFmpeg:',imageio_ffmpeg.get_ffmpeg_exe())
    # Exercise only renderer initialization; does not create a video or solve poses.
    if c['recording']['physics_video'] or c['recording']['trajectory_video']:
        with mujoco.Renderer(e.model,height=c['recording']['height'],width=c['recording']['width']) as renderer:
            renderer.update_scene(e.data)
            assert renderer.render().shape==(c['recording']['height'],c['recording']['width'],3)
    print('Preflight passed; no mj_step executed. This is not a derivation validation.')


if __name__=='__main__':main()
