import torch

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.interpolate import Rbf
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from itertools import product

class MLP(torch.nn.Module):
    def __init__(self, dim_list, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        net = [torch.nn.Linear(dim_list[0] + (1 if time_varying else 0), dim_list[1]),
               torch.nn.SELU()]

        for i, (dim_in, dim_out) in enumerate(zip(dim_list[1:-2], dim_list[2:-1])):
            net.append(torch.nn.Linear(dim_in, dim_out))
            net.append(torch.nn.SELU())
        net.append(torch.nn.Linear(dim_list[-2], dim_list[-1]))
        self.net = torch.nn.Sequential(*net)
        
    def forward(self, x):
        dx = self.net(x)
        return dx

class Scaler:
    def transform(self, x):
        x += 1e-6
        return np.where(x<1, np.log(x)+1, x)
    def inverse_transform(self, x):
        x -= 1e-6
        return np.where(x<1, np.exp(x-1), x)

def cal_epr(ref_net, preds):
    """
    Calculate the Edge Precision Rate (EPR) for the inferred network.
    Args:
        ref_net (np.ndarray): The reference network, a binary matrix of shape (n_genes, n_genes).
        preds (np.ndarray): The predicted network, a confidence matrix of shape (n_genes, n_genes).
    Returns:
        float: The Early Precision Rate (EPR) value.
    """
    return len(set(np.argpartition(preds.reshape(-1), -ref_net.sum())[-ref_net.sum():]).intersection(set(np.where(ref_net.reshape(-1))[0]))) / ref_net.mean() / ref_net.sum()

def plot_model(
    ax,
    data,
    traj,
    sf2m_model,
    sf2m_score_model,
    plot_traj=True,
    plot_sf2m_model=True,
    plot_sf2m_score_model=True,
    title="",
    grid_num=10,
    time_interpolation_method="knn",
):
    """
    Generates a comprehensive plot including data scatter, trajectories, and vector fields.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        data (list): A list of numpy arrays, where each array is a snapshot of gene expressions.
        traj (np.ndarray): Trajectory data.
        sf2m_model (torch.nn.Module): The primary vector field model.
        sf2m_score_model (torch.nn.Module): The score-based vector field model.
        plot_traj (bool): Whether to plot the trajectories.
        plot_sf2m_model (bool): Whether to plot the vector field for sf2m_model.
        plot_sf2m_score_model (bool): Whether to plot the vector field for sf2m_score_model.
        title (str): The title for the plot.
        grid_num (int): The number of grid points for the vector field.
        time_interpolation_method (str): Method to interpolate time for grid points ('knn', 'rbf').
    """
    # --- 1. Data Preparation ---
    device = next(sf2m_model.parameters()).device
    n_gene = data[0].shape[1]
    t = np.concatenate([np.ones((len(snapshot), 1)) * i for i, snapshot in enumerate(data)], axis=0)
    data_all = np.concatenate(data)
    scaler = Scaler()

    # --- 2. Dimensionality Reduction ---
    np.random.seed(0)
    dr_model = PCA(n_components=2, random_state=0)
    dr_model.fit(data_all)
    data_all_dr = dr_model.transform(data_all)

    # --- 3. Scatter Plot of Data Points ---
    idx_count = 0
    cmap = plt.get_cmap("gist_rainbow")
    cmap_colors = cmap(np.linspace(0, 1, len(data)))
    for i, snapshot in enumerate(data):
        snapshot_dr = dr_model.transform(snapshot)
        ax.scatter(snapshot_dr[:, 0], snapshot_dr[:, 1], s=1, alpha=0.5, label=f"t={i}", color=cmap_colors[i])
        idx_count += len(snapshot_dr)

    # --- 4. Plot Trajectories (Optional) ---
    if plot_traj:
        traj_dr = dr_model.transform(traj.reshape(-1, n_gene)).reshape(*traj.shape[:-1], 2)
        for i in range(len(data)):
            # Find corresponding indices for this time point
            start_idx = sum(len(s) for s in data[:i])
            end_idx = start_idx + len(data[i])
            # Select a few trajectories to plot
            num_trajs_to_plot = min(5, end_idx - start_idx)
            plot_indices = np.random.choice(range(start_idx, end_idx), size=num_trajs_to_plot, replace=False)
            ax.plot(traj_dr[:, plot_indices, 0], traj_dr[:, plot_indices, 1], color=cmap_colors[i], alpha=1, linewidth=0.2)

    # --- 5. Adjust Axes Limits ---
    xlims = list(ax.get_xlim())
    ylims = list(ax.get_ylim())
    data_span = data_all_dr.ptp(axis=0)
    xylim_extension = 0.25
    max_xlims = (data_all_dr[:,0].min() - xylim_extension * data_span[0], data_all_dr[:,0].max() + xylim_extension * data_span[0])
    max_ylims = (data_all_dr[:,1].min() - xylim_extension * data_span[1], data_all_dr[:,1].max() + xylim_extension * data_span[1])
    if xlims[0] < max_xlims[0]:
        xlims[0] = max_xlims[0]
    if xlims[1] > max_xlims[1]:
        xlims[1] = max_xlims[1]
    if ylims[0] < max_ylims[0]:
        ylims[0] = max_ylims[0]
    if ylims[1] > max_ylims[1]:
        ylims[1] = max_ylims[1]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    if plot_sf2m_model or plot_sf2m_score_model:
    # --- 6. Vector Field Grid Generation ---
        xs = np.linspace(*ax.get_xlim(), grid_num)
        ys = np.linspace(*ax.get_ylim(), grid_num)
        Zg = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)

        hull = ConvexHull(data_all_dr)
        hull_path = mpath.Path(data_all_dr[hull.vertices])
        Zg = Zg[hull_path.contains_points(Zg)]

    # --- 7. Time Interpolation for Grid Points ---
        if time_interpolation_method == "knn":
            time_interpolation = KNeighborsRegressor(n_neighbors=5, weights='distance')
            time_interpolation.fit(data_all_dr, t)
            tg = time_interpolation.predict(Zg).reshape(-1, 1)
        elif time_interpolation_method == "rbf":
            time_interpolation = Rbf(data_all_dr[:, 0], data_all_dr[:, 1], t.ravel(), function='thin_plate')
            tg = time_interpolation(Zg[:, 0], Zg[:, 1]).clip(0, len(data) - 1).reshape(-1, 1)
        else:
            raise ValueError("Unknown time interpolation method")

        X_high_dim = dr_model.inverse_transform(Zg).clip(0, None)

    # --- 8. Helper function for Quiver Plot ---
        def _plot_quiver(vec_model, color="black", label=None):
            W = dr_model.components_
            
            def project_velocity(dx):
                v = (W @ dx.T).T
                if dr_model.whiten:
                    v /= np.sqrt(dr_model.explained_variance_[:2])
                return v

            with torch.no_grad():
                dX_high_dim = vec_model(torch.tensor(np.hstack([scaler.transform(X_high_dim), tg])).float().to(device)).cpu().numpy()

            V2 = project_velocity(dX_high_dim)
            Zx, Zy = Zg[:, 0], Zg[:, 1]
            Ux, Uy = V2[:, 0], V2[:, 1]

            ax_x_len = ax.get_xlim()[1] - ax.get_xlim()[0]
            ax_y_len = ax.get_ylim()[1] - ax.get_ylim()[0]
            L_max = 0.8 * np.hypot(ax_x_len / 50, ax_y_len / 50)
            mag = np.median(np.hypot(Ux, Uy))
            scale = L_max / (mag + 1e-6)

            ax.quiver(Zx, Zy, Ux * scale, Uy * scale, angles='xy', scale_units='xy', scale=1, width=0.002, color=color, label=label)

    # --- 9. Plot Vector Fields (Optional) ---
        if plot_sf2m_model:
            _plot_quiver(sf2m_model, color="black", label="$v(\\theta)$")
        if plot_sf2m_score_model:
            _plot_quiver(sf2m_score_model, color="blue", label="$s(\\theta)$")

    # --- 10. Final Touches ---
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
