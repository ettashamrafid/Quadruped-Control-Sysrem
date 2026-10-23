# Data dictionary

## Common conventions

Positions: mm relative to fixed FL_O with world-aligned axes, Z upward. Angles in CSV: degrees; NPZ qpos uses MuJoCo radians and its model order, not just independent joints. qvel: rad/s. Independent order: FL_abad, FL_a, FL_b. Passive joints: FL_passive_B, FL_passive_C. Boolean CSV values are True/False. Missing comparison values are blank; unavailable NPZ states are NaN and have available=False.

## Each stage: samples.csv

- sample: trajectory index; excludes approach.
- nominal_time_s: parameter time of desired trajectory.
- simulation_time_s: accumulated relaxation time, when a pose exists.
- target_mm_x/y/z: requested trajectory point.
- measured_FL_*_deg: actual independent joint coordinates.
- actual_mm_x/y/z: measured foot position.
- position_error_mm: actual versus target Euclidean distance.
- requested_FL_*_deg: analytical phase angle requests.
- ik_fk_self_error_mm: analytical FK of requested IK angles versus target.
- ok: engine diagnostics passed; does not assert analytical foot tracking.
- status: VALID, INVALID_POSE or IK_NO_SOLUTION.
- error: analytical failure text when applicable.
- tracking: mm in baseline target mode; degrees in analytical angle mode.
- closure_mm: gap between two D sites.
- max_joint_speed_rad_s: maximum absolute qvel.
- steps: relaxation steps used at this point.
- home_assembly_branch, assembly_measure_m2: assembly-side sign check and signed cross-product measure.
- settled: speed criterion satisfied.
- contacts, penetration_mm: zero in this contacts-off workflow.
- limit_violation_deg: largest defined-limit violation; no supplied limits means this does not establish physical feasibility.

## Each stage: states.npz

qpos, qvel: full state arrays at each sampled final pose. position_mm: measured XYZ. valid: engine validity. available: a measured pose exists, even if invalid. time_s: nominal trajectory time. target_mm: requested XYZ. signature: source/config/environment identifier. These arrays are pose evidence, not full solver restart checkpoints.

## physics_frames.npz

simulation_time_s, sample, target_mm, qpos and qvel at captured video-frame states. sample=-1 denotes baseline approach. The target-versus-angle constraint mode follows the phase. Initial and final states may fall between frame times; all final sample states remain in states.npz.

## comparison/comparison.csv

- baseline_valid, analytical_pose_valid, both_valid: measurement eligibility.
- baseline_tracking_error_mm: baseline versus target.
- fk_error_mm: analytical FK(measured baseline angles) versus baseline XYZ.
- ik_target_error_mm: analytical angle-driven measured foot versus desired XYZ.
- two_run_position_error_mm: measured analytical-run versus baseline foot.
- fk_at_ik_measured_error_mm: analytical FK at measured analytical-run angles versus that run's foot.
- J_analytical_rc_mm_rad, J_cad_rc_mm_rad: matrix entries, rows XYZ and columns independent joint order.
- jacobian_max_entry_error_mm_rad: largest absolute entry difference.
- closure_velocity_residual_m_rad: maximum residual after passive-rate elimination.
- jacobian_reference_valid: baseline valid and residual < 1e-7.
- passive_condition, analytical_condition: condition numbers; singular/invalid values are blank, not zero.
- velocity_analytical_mm_s_x/y/z, velocity_cad_mm_s_x/y/z: J times the configured independent rates.
- velocity_error_mm_s: norm of difference of those predictions.
- baseline_FL_*_deg, analytical_measured_FL_*_deg: measured angles from each run.
- angle_difference_FL_*_deg: analytical-run minus baseline angle, wrapped to [-180,180).
- failure fields: analytical evaluation or singularity error messages.

Component differences (baseline_minus_target_mm_*, analytical_minus_target_mm_*, analytical_minus_baseline_mm_*, fk_minus_baseline_mm_*) retain the signed XYZ errors.

## summary.json

metrics contains valid_only and all_finite_diagnostic_only groups. Each reports count, maximum, RMS, median and p95. Missing groups have count=0 and null values. acceptance checks require full eligible coverage as well as a maximum-error threshold. Stage completion and comparison acceptance are separate concepts. stage_diagnostics includes all-finite closure/speed/tracking/step statistics and failure counts for each phase.

## Integrity

manifest.json records source/mesh SHA-256 hashes, config and exact environment. stage status.json contains completion state and sample/state/video hashes. Comparison checks completed stages and numerical artifact hashes. The post-run audit independently recomputes key saved-data relationships and valid-only summaries.
