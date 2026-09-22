# Minimal sphere IMD tutorial

Open **[imd_tutorial.ipynb](imd_tutorial.ipynb)** and run the cells in order:

1. Sample or load uniform/von Mises–Fisher observations on the unit sphere.
2. Build and visualize a radius proximity graph.
3. Compute graph drift and carré-du-champ covariance.
4. Run and visualize six continuous Euler–Maruyama trajectories.

```bash
cd imd_tutorial
python -m pip install -r requirements.txt
python -m jupyterlab imd_tutorial.ipynb
```

Use the same Python environment for installation and the notebook kernel.
Numerical code needs only NumPy and SciPy; plots use Plotly. No Torch, training,
main-repository imports, or external data are needed. The notebook caches small
point clouds in `data/`; deleting that directory simply regenerates them.

## Source files

- `imd_tutorialsrc/datasets.py`: uniform and von Mises–Fisher sphere samplers.
- `imd_tutorialsrc/graph.py`: binary radius graph, density normalization, drift and CDC.
- `imd_tutorialsrc/dynamics.py`: off-node interpolation and Euler–Maruyama.
- `imd_tutorialsrc/plotting.py`: sphere-only adaptation of `plot_diffusion_3D` from
  `/Users/victorkawasaki-borruat/Desktop/code/implicit-manifolds/implicit-manifolds/src/plotting.py`.
  Point/vector/trajectory rendering is preserved; unrelated mesh modes are removed
  to avoid dependencies on the sibling repository. All notebook plots use this helper.

The binary radius kernel is scaled to target generator **½Δ** on a two-dimensional
surface. `alpha=1` corrects for observation density asymptotically. The CDC array
is the full noise covariance (no extra factor ½). The simulator discards covariance
eigenvalues below 1% of the largest by default and never projects paths to the sphere.
Finite-cloud and time-discretization errors therefore remain visible.
