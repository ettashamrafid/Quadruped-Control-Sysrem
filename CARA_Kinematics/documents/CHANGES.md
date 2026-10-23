# Changes from the supplied CARA_comparison package

- Contacts-off is the only supported setting. The CAD model and original mesh assets are retained.
- Trajectory samples: 401 -> 1601. Geometry, durations and analytical equations are unchanged.
- Replaced the combined original runner with workflow.py: baseline -> saved analytical run -> offline comparison.
- Both phases store samples.csv and full qpos/qvel states with explicit validity and availability flags.
- Added physics MP4 capture plus a separate, clearly labeled nominal-time trajectory playback for each phase.
- Added automatic PDF/Markdown team report, separate failure records, per-entry Jacobians, joint-angle plots, valid-only and diagnostic statistics, full-coverage thresholds.
- Added environment/source/config signatures, artifact hashes, checkpointing, completion status and overwrite guards.
- Fixed mocap target indexing, reference-aware joint equality offsets, and degree/radian handling for optional limits. Added explicit fixed-base five-hinge compatibility checks and offscreen framebuffer dimensions.
- Added local preflight and saved-result audit commands. Revised dependencies are pinned; the revised runtime workflow was NOT executed during preparation.
- Kept analytical.py, original_calculator.py and source_derivations.pdf byte-for-byte from the supplied source. No new fit or rederivation was performed.
- Excluded all old generated result folders, old reports, obsolete batch launchers, caches, duplicated legacy tests and machine-specific generated XML snapshots. The original uploaded archive remains unchanged.
- Mesh assets were conservatively retained in full so model dependencies are not accidentally removed.

