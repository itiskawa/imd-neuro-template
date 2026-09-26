"""
Resumes heavy_run.py's diagnostic suite using the already-saved checkpoint and
simulated trajectories, skipping the (already-completed) training/simulation.
"""
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

import heavy_run as hr

REPO_ROOT = hr.REPO_ROOT

data = np.load(REPO_ROOT / "data" / "simulated" / "heavy_head_direction_trajectories.npz")
hidden_states = data["hidden_states"]
theta = data["theta"]
hr.log(f"loaded saved simulation: hidden_states shape {hidden_states.shape}")

summary_lines = ["Heavy run summary (diagnostics resumed after interrupted first attempt)",
                  "=" * 70, ""]
summary_lines.append(f"Config: n_steps={hr.N_STEPS}, seq_len={hr.SEQ_LEN}, lr={hr.LR}, hidden_size={hr.HIDDEN_SIZE}")
summary_lines.append("Final training loss: 0.00089 (from original log)")
summary_lines.append("Sanity check max angular error: 0.200 rad (from original log)")
summary_lines.append(f"Simulated {hr.N_SEQUENCES_SIM} trajectories of length {hr.SEQ_LEN}")

t_start = time.time()
h_flat = theta_flat = None
try:
    h_flat, theta_flat, evr = hr.pca_plots(hidden_states, theta)
    summary_lines.append(f"PCA explained variance (top 3): {evr} (cumulative {evr.sum():.3f})")
except Exception:
    hr.log("pca_plots failed:\n" + traceback.format_exc())
    summary_lines.append("PCA: FAILED, see log")

if h_flat is not None:
    try:
        homology_results = hr.persistent_homology_with_stability(h_flat, theta_flat)
        summary_lines.append(f"Top 5 H1 persistence values: {homology_results['top_h1']}")
        summary_lines.append(f"H0 infinite-persistence components (should be 1): {homology_results['n_inf_h0']}")
        summary_lines.append(f"Top H2 persistence values (should be near 0): {homology_results['top_h2']}")
        summary_lines.append(
            f"Bootstrap stability ({hr.N_BOOTSTRAP} resamples of {hr.BOOTSTRAP_SAMPLE_SIZE} points): "
            f"max H1 per resample={homology_results['bootstrap_top_h1']}, "
            f"mean={homology_results['bootstrap_top_h1'].mean():.4f}, "
            f"std={homology_results['bootstrap_top_h1'].std():.4f}"
        )
    except Exception:
        hr.log("persistent_homology_with_stability failed:\n" + traceback.format_exc())
        summary_lines.append("Persistent homology: FAILED, see log")

    try:
        circ_coord = hr.circular_coords_aggregate(h_flat, theta_flat)
        if circ_coord is not None:
            summary_lines.append("Circular coordinates: saved (see heavy_circular_coords.png)")
        else:
            summary_lines.append("Circular coordinates: FAILED (see log), skipped")
    except Exception:
        hr.log("circular_coords_aggregate failed:\n" + traceback.format_exc())
        summary_lines.append("Circular coordinates: FAILED, see log")

elapsed = time.time() - t_start
summary_lines.append(f"\nDiagnostics runtime: {elapsed/60:.1f} minutes")
summary_lines.append("\nFigures saved in results/figures/ (whichever succeeded):")
summary_lines.append("  heavy_loss_curve.png, heavy_sanity_check.png, heavy_pca3d.png,")
summary_lines.append("  heavy_persistence_diagram.png, heavy_circular_coords.png")

with open(hr.SUMMARY_PATH, "w") as f:
    f.write("\n".join(summary_lines) + "\n")

hr.log(f"DONE. Diagnostics runtime: {elapsed/60:.1f} min. Summary written to {hr.SUMMARY_PATH}")
