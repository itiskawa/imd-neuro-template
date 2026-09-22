"""Sphere-only adaptation of the requested plot_diffusion_3D helper.

Source: implicit-manifolds/implicit-manifolds/src/plotting.py, copied 2026-09-18.
Point, vector, and trajectory rendering are preserved; unrelated mesh modes
are removed so this directory needs no external repository or mesh loader.
The original function returns a Plotly figure; call fig.show() in a notebook.
"""

from typing import Optional
import numpy as np
import plotly.graph_objects as go

def add_vector_lines(fig, 
                     points, 
                     vectors, 
                     color, name, scale=1.0):
    """
    Helper to draw vectors as lines.
    Each vector is drawn from P to (P + scale * V).
    """
    end_points = points + (vectors * scale)

    x_lines = np.empty(3 * len(points))
    y_lines = np.empty(3 * len(points))
    z_lines = np.empty(3 * len(points))

    x_lines[0::3] = points[:, 0]
    x_lines[1::3] = end_points[:, 0]
    x_lines[2::3] = None

    y_lines[0::3] = points[:, 1]
    y_lines[1::3] = end_points[:, 1]
    y_lines[2::3] = None

    z_lines[0::3] = points[:, 2]
    z_lines[1::3] = end_points[:, 2]
    z_lines[2::3] = None

    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode="lines",
        line=dict(color=color, width=5),
        name=name,
        hoverinfo="none",
    ))


def plot_diffusion_3D(
    X: np.ndarray,
    mode: str = None,
    score: Optional[np.ndarray] = None,
    vector_scale: float = 1,
    point_values: Optional[np.ndarray] = None,
    trajectories: Optional[list[np.ndarray] | np.ndarray] = None,
    trajectory_style: Optional[dict | list[dict]] = None,
    marker_style: Optional[dict] = None,
    show_scene : bool = True,
    show_legend : bool = True
):
    fig = go.Figure()

    # points
    marker = (marker_style.copy() if marker_style is not None else dict(size=2, color="gray", opacity=0.6))
    if point_values is not None:
        point_values = np.asarray(point_values)
        if point_values.ndim != 1 or point_values.shape[0] != X.shape[0]:
            raise ValueError("point_values must be a 1D array of length N (same as X).")
        marker.update(dict(
            color=point_values,
            colorscale=marker.get("colorscale", "magma"),
            colorbar=marker.get("colorbar", dict(title="")),
        ))
        marker.setdefault("cmin", float(np.nanmin(point_values)))
        marker.setdefault("cmax", float(np.nanmax(point_values)))
    fig.add_trace(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        opacity=0.6,
        mode="markers",
        marker=marker,
        name="points",
    ))

    # The tutorial only needs point clouds on the sphere, not external meshes.
    if mode not in (None, "sphere"):
        raise ValueError("This lightweight tutorial copy supports mode=None or 'sphere'.")

    if score is not None:
        add_vector_lines(fig, X, score, "red", "scaled score", scale=vector_scale) 
        print("X shape:", X.shape)
        print("score shape:", score.shape)
        print("X finite:", np.isfinite(X).all())
        print("score finite:", np.isfinite(score).all())
        print("score min:", np.nanmin(score))
        print("score max:", np.nanmax(score))
        print("score norm min:", np.nanmin(np.linalg.norm(score, axis=1)))
        print("score norm max:", np.nanmax(np.linalg.norm(score, axis=1)))
        print("score norm mean:", np.nanmean(np.linalg.norm(score, axis=1)))
    if trajectories is not None:
        traj_color_cycle = ["darkblue","red","green", "purple", "orange", "brown", "magenta", "cyan", "gold", "navy"]
        traj_list = trajectories if isinstance(trajectories, (list, tuple)) else [trajectories]
        style_list = None
        if isinstance(trajectory_style, (list, tuple)):
            style_list = trajectory_style
        for i, traj in enumerate(traj_list):
            traj_arr = np.asarray(traj, float)
            if traj_arr.ndim != 2 or traj_arr.shape[1] != 3:
                raise ValueError("Each trajectory must be a (T,3) array.")
            if style_list is not None:
                style = style_list[i] if i < len(style_list) else style_list[-1]
            else:
                style = trajectory_style or dict(color=traj_color_cycle[i % len(traj_color_cycle)], width=4)
            color = style.get("color", traj_color_cycle[i % len(traj_color_cycle)])
            fig.add_trace(go.Scatter3d(
                x=traj_arr[:, 0],
                y=traj_arr[:, 1],
                z=traj_arr[:, 2],
                mode="lines+markers",
                line=style,
                marker=dict(size=1, color=color),
                name=f"trajectory_{i}",
            ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=show_scene),
            yaxis=dict(visible=show_scene),
            zaxis=dict(visible=show_scene),
            aspectmode="data",
        ),
        showlegend=show_legend,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    r=0.8
    fig.update_layout(
    scene_camera=dict(
        #eye=dict(x=-0.5*r, y=1.2*r, z=0.7*r)
        eye=dict(x=1*r, y=1*r, z=1*r) # Torus
        )
    )
    return fig
