# CARA: contacts-off trajectory validation

Run this package locally to generate them. The analytical FK, IK, Jacobian and original calculator are unchanged from the supplied project. The comparison workflow and recording support were revised.

Read `documents/TEAM_GUIDE.pdf` or `documents/TEAM_GUIDE.md` first.

## Windows setup (PowerShell or CMD)

Use Python 3.12 and extract the entire ZIP. Open a terminal in this project directory.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python check_setup.py
```

The package versions are pinned for repeatability. This revised package has received static checks only; renderer, encoder and end-to-end runtime checks are left for your local execution. Preflight checks OpenGL rendering and locates FFmpeg without stepping the physics. A working graphics driver is needed even when the viewer is hidden, because MP4 recording renders frames.

## Run all three stages

```powershell
.venv\Scripts\python workflow.py --stage all --out results\run_1601 --preview
```

Keep the viewer open until each phase finishes. The baseline viewer closes automatically before the analytical viewer opens. Closing early marks the stage incomplete. There is no final infinite replay loop.

To hide the live viewer while keeping recordings, omit `--preview`. `run_all.bat` uses `python` from your activated environment; the explicit `.venv` commands above avoid interpreter ambiguity.

## Run the stages separately

Use the same output directory, source files, configuration and Python environment throughout:

```powershell
.venv\Scripts\python workflow.py --stage baseline --out results\run_1601 --preview
.venv\Scripts\python workflow.py --stage analytical --out results\run_1601 --preview
.venv\Scripts\python workflow.py --stage compare --out results\run_1601
.venv\Scripts\python tests\test_recorded_results.py results\run_1601
```

The comparison stage reads saved states and does not step the simulation. Completed or partial stage folders are never silently overwritten. If a stage is interrupted, preserve its evidence and start a fresh run directory; automatic resume is not implemented. Change the folder name for each experiment.

## Defaults

- Contacts are forcibly OFF, including when constructing `CadEngine` directly.
- 1,601 trajectory samples, up from 401; nominal target spacing is 0.01 s over 16 s.
- Center (-80, 65, -240) mm; radius 35 mm; rise 60 mm; one base circle plus four rising spiral turns.
- Independent joint order: FL_abad, FL_a, FL_b; internal angles in radians, CSV angles in degrees.
- Positions relative to fixed FL_O, world-aligned axes, Z upward.
- Gravity zero; 1 ms physics step; numerical damping 0.1; up to 1,000 relaxation steps per point.
- MP4 recording enabled at 960 x 720, 30 fps for both phases.
- Physical joint limits have not been supplied. More samples improve trajectory coverage, not physical realism or proof of global correctness.

## Outputs

Under your chosen output directory:

| File / folder | Purpose |
| --- | --- |
| manifest.json | Source/asset hashes, exact environment, config, timestamp |
| used_config.json, trajectory.csv | Exact requested experiment |
| baseline/samples.csv | Desired XYZ, measured angles/XYZ, errors, diagnostics |
| baseline/states.npz | Every final full qpos/qvel, XYZ, validity/availability |
| baseline/physics_settling.mp4 | Actual transient simulation capture |
| baseline/physics_frames.npz | Joint states and times for captured physics frames |
| baseline/trajectory_playback.mp4 | Nominal-time saved-pose playback |
| analytical/ | Same files for analytical IK angle-driven run |
| comparison/comparison.csv | Joined errors, joint differences, Jacobian entries and velocities |
| comparison/summary.json | Valid-only and all-finite metrics, coverage and acceptance |
| comparison/failures.csv | Samples failing validity or configured comparisons |
| comparison/paths.png, errors.png, joint_angles.png | Figures |
| comparison/team_results_report.pdf, .md | Automatically generated team report |

`status.json` is written in each phase. Only complete phase files with matching signatures and hashes are accepted for comparison. Invalid samples are retained, never converted into a success. Partial recordings may exist after an interruption but are not a completed test.

## Recording meaning

`physics_settling.mp4` captures actual simulated states every 33 physics steps at the default settings and includes baseline approach. At 30 fps its playback speed is about 0.99 times simulation time. Captions show simulated time and sample index. It does not capture the desktop screen.

`trajectory_playback.mp4` uses nearest saved sampled poses at nominal trajectory times; it is a separate approximately 16-second animation. It has no new dynamics integration. All 1,601 numerical samples are saved, but not every sample appears in a 30 fps video. Missing IK solutions are labeled and retain the last display pose; data arrays mark them unavailable.

Blue is the desired path, green is the achieved trail, and the red target marker is visual only during the analytical run. Video encoding/render failures fail the stage rather than silently dropping the requested recording. If you deliberately disable recording in config, the stage can run without those video files and records that configuration.

## Repository contents and provenance

The flat source layout preserves portable imports. `cara_one_leg.xml` and `meshes/` are the portable model; all original mesh assets are retained. `analytical.py` calls `original_calculator.py`; neither was refitted. `cad_engine.py` maintains the closed loop and switches between target and angle modes. `workflow.py` saves both runs before computing comparisons; `recording.py` handles videos; `report.py` generates results documentation.

`documents/source_derivations.pdf` is the supplied derivation reference, not a new rederivation. See `documents/CHANGES.md` and `documents/DATA_DICTIONARY.md`. 