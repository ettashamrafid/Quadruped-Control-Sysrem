"""MP4 capture of actual settling states plus nominal-time saved-pose playback."""
import time
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw


class Capture:
    def __init__(self, engine, targets, config, folder, label, preview=False):
        self.e, self.targets, self.config = engine, targets, config
        self.folder, self.label = Path(folder), label
        self.viewer = self.renderer = self.writer = None
        self.target = targets[0].copy()
        self.index = -1
        self.trail = []
        self.frames = []
        self.steps = 0
        rc = config['recording']
        self.fps = rc['fps']
        self.stride = max(1, round(1 / (self.fps * engine.model.opt.timestep)))
        self.physics_frames = 0
        self.last_preview = 0.0
        try:
            if preview:
                import mujoco.viewer as viewer
                self.viewer = viewer.launch_passive(engine.model, engine.data)
                self.configure_camera(self.viewer.cam)
            if rc['physics_video'] or rc['trajectory_video']:
                self.renderer = mujoco.Renderer(engine.model, height=rc['height'], width=rc['width'])
                self.camera = mujoco.MjvCamera()
                self.configure_camera(self.camera)
                if rc['physics_video']:
                    self.writer = self.open_writer(self.folder / 'physics_settling.mp4', self.fps)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def open_writer(path, fps):
        import imageio.v2 as imageio
        return imageio.get_writer(str(path), fps=fps, codec='libx264', macro_block_size=2,
                                  ffmpeg_log_level='error', quality=8)

    def configure_camera(self, camera):
        points = self.e.origin + self.targets / 1000
        camera.lookat[:] = (points.min(axis=0) + points.max(axis=0)) / 2 + [0, 0, .06]
        camera.distance = .7
        camera.azimuth = 125
        camera.elevation = -15

    def refresh(self):
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            return False
        now = time.perf_counter()
        if now - self.last_preview >= 1 / 30:
            with self.viewer.lock():
                self.viewer.user_scn.ngeom = 0
                self.draw_overlay(self.viewer.user_scn)
            self.viewer.sync()
            self.last_preview = now
        return True

    def set_sample(self, index, target):
        self.index, self.target = index, np.asarray(target).copy()

    def on_step(self):
        self.steps += 1
        if self.writer is not None and self.steps % self.stride == 0:
            self.frames.append((self.e.data.time, self.index, self.target.copy(),
                                self.e.data.qpos.copy(), self.e.data.qvel.copy()))
            self.writer.append_data(self.image('Actual constraint relaxation; simulation time %.3f s' % self.e.data.time))
            self.physics_frames += 1

    def draw_overlay(self, scene):
        """Same world-coordinate overlays for live viewer and recorded frames."""
        def sphere(position, color, radius):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.full(3, radius), position, np.eye(3).ravel(),
                               np.array(color, dtype=float))
            geom.emission = 0.8
            scene.ngeom += 1

        def segment(a, b, color, width):
            if scene.ngeom >= scene.maxgeom or np.linalg.norm(a-b) < 1e-12:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                               np.zeros(3), np.eye(3).ravel(), np.array(color, dtype=float))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
            geom.emission = 0.8
            scene.ngeom += 1

        # Entire requested path is visible from the beginning. Actual trail grows.
        for points, color, width in [(self.targets, [.05,.65,1,1], .0015),
                                     (np.asarray(self.trail), [.05,1,.25,1], .0018)]:
            if len(points) < 2:
                continue
            ids = np.unique(np.linspace(0, len(points)-1, min(500, len(points))).astype(int))
            world = self.e.origin + points / 1000
            for a,b in zip(ids[:-1],ids[1:]):
                segment(world[a], world[b], color, width)
        foot = self.e.data.site('FL_P').xpos.copy()
        if self.trail:
            segment(self.e.origin + np.asarray(self.trail[-1])/1000,
                    foot, [.05,1,.25,1], .0018)
        sphere(self.e.origin + self.target/1000, [1,.1,.1,1], .004)
        sphere(foot, [.05,1,.25,1], .003)
        sphere(self.e.origin + self.targets[0]/1000, [1,.65,.05,1], .0035)
        sphere(self.e.origin + self.targets[-1]/1000, [1,.1,1,1], .004)

    def image(self, caption):
        self.renderer.update_scene(self.e.data, camera=self.camera)
        self.draw_overlay(self.renderer.scene)
        im = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, im.width, 78), fill=(15,25,35))
        draw.text((12,8), self.label + ' | contacts OFF | sample ' + str(self.index), fill='white')
        draw.text((12,29), caption, fill='white')
        draw.text((12,50), 'Blue: desired path   Green: achieved path   Red marker: target', fill='white')
        return np.asarray(im)

    def trajectory(self, states, times, targets, positions, valid, available):
        """Nearest saved-pose animation: deliberately not a physical-time claim."""
        if not self.config['recording']['trajectory_video']:
            return
        self.trail = []
        cursor = -1
        with self.open_writer(self.folder / 'trajectory_playback.mp4', self.fps) as writer:
            frame_times = np.linspace(float(times[0]),float(times[-1]),round((times[-1]-times[0])*self.fps)+1)
            for t in frame_times:
                i = int(np.argmin(np.abs(times-t)))
                self.set_sample(i, targets[i])
                while cursor < i:
                    cursor += 1
                    if available[cursor]:
                        self.trail.append(positions[cursor].copy())
                if available[i]:
                    self.e.data.qpos[:] = states[i]
                    self.e.data.qvel[:] = 0
                    self.e.data.mocap_pos[self.e.target_mocap] = self.e.origin + targets[i]/1000
                    mujoco.mj_forward(self.e.model,self.e.data)
                    status = 'VALID' if valid[i] else 'INVALID POSE'
                else:
                    status = 'NO SOLUTION: displayed pose is last available state'
                writer.append_data(self.image('Saved-pose playback; nominal t=%.3f s | %s' % (t,status)))

    def close(self):
        try:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
        finally:
            if self.frames:
                np.savez_compressed(self.folder/'physics_frames.npz',
                    simulation_time_s=np.array([x[0] for x in self.frames]),
                    sample=np.array([x[1] for x in self.frames]),
                    target_mm=np.array([x[2] for x in self.frames]),
                    qpos=np.array([x[3] for x in self.frames]),
                    qvel=np.array([x[4] for x in self.frames]))
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
