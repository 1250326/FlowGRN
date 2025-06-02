import argparse
from pathlib import Path
from anndata import AnnData
import numpy as np
import scanpy
import scipy
from tqdm import tqdm
import pandas as pd
import pickle
import ot
import warnings

import torch
from torchdyn.core import NeuralODE
import torchcfm
from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from torchcfm.utils import torch_wrapper
from torchcfm.optimal_transport import OTPlanSampler

parser = argparse.ArgumentParser()
parser.add_argument('--expr-fn', type=str, help="Path of the ExpressionData.csv")
parser.add_argument('--pseudoT-fn', type=str, help="Path of the Pseudotime.csv")
parser.add_argument('--out-dir', type=str, help="Output directory to save the results")
parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
parser.add_argument('--branch-id', type=int, default=0, help="Branch ID to process (0-indexed)")
parser.add_argument('--n-snapshot', type=int, default=5, help="Number of snapshots to batch by if not using precomputed batches")
parser.add_argument('--precomputed-batch-fn', type=str, default=None, help="Path to precomputed batch file")
parser.add_argument('--no-knn', action='store_true', default=False, help="Disable kNN graph distance")
parser.add_argument('--no-mask-l1', action='store_true', default=False, help="Disable dropout-robust cell similarity measure")
args = parser.parse_args()

# Ensure output directory exists
args.out_dir = Path(args.out_dir)
args.out_dir.mkdir(parents=True, exist_ok=True)
branch_id = int(args.branch_id)
args.out_dir = args.out_dir / str(branch_id)
args.out_dir.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Check if CUDA is available and set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Prepare the batched input data
if args.precomputed_batch_fn is not None:
    # Load precomputed batches
    with open(args.precomputed_batch_fn, 'rb') as f:
        gene_expression_list = pickle.load(f)
    n_gene = gene_expression_list[branch_id][0].shape[1]
    gene_expression = np.concatenate(gene_expression_list[branch_id])
    gene_expression = pd.DataFrame(gene_expression)
    n_cell = gene_expression.shape[0]
    n_snapshot = len(gene_expression_list[branch_id])
    n_time = n_snapshot
else:
    # If no precomputed batches, load data and create batches
    gene_expression = pd.read_csv(args.expr_fn, index_col=0, header=0)
    gene_expression = gene_expression.T
    gene_list = gene_expression.columns.tolist()

    # Load pseudotime data
    time_df = pd.read_csv(args.pseudoT_fn, index_col=0, header=0, na_values='NA')
    pseudotime = time_df.fillna(0).values.sum(1)
    n_cell, n_gene = gene_expression.shape
    
    n_snapshot = args.n_snapshot

    n_branch = time_df.shape[1]
    assert branch_id < n_branch

    # Batch gene expression data by branch and time
    gene_expression_list = []
    time_split = np.linspace(0, pseudotime.max(), n_snapshot+1)
    time_split[-1] += 1e-2
    for i in range(n_branch):
        branch_list = []
        for t in range(n_snapshot):
            branch_list.append(gene_expression.loc[time_df.iloc[(pseudotime>=time_split[t])&(pseudotime<time_split[t+1]), i].dropna().index,:].values)
        gene_expression_list.append(branch_list)

def get_batch(FM, X, batch_size, n_times, return_noise=False):
    """Construct a batch with points from each timepoint pair"""
    # copied from https://github.com/atong01/conditional-flow-matching/blob/main/examples/single_cell/single-cell_example.ipynb
    ts = []
    xts = []
    uts = []
    noises = []
    for t_start in range(n_times - 1):
        x0 = (
            torch.from_numpy(X[t_start][np.random.randint(X[t_start].shape[0], size=batch_size)])
            .float()
            .to(device)
        )
        x1 = (
            torch.from_numpy(
                X[t_start + 1][np.random.randint(X[t_start + 1].shape[0], size=batch_size)]
            )
            .float()
            .to(device)
        )
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise
            )
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    return t, xt, ut

class MLP(torch.nn.Module):
    """A simple MLP model for flow matching and score matching."""
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

def masked_l1_mean_distance(a, b):
    """
    Compute the cell similarity measure.

    Parameters
    ----------
    a : numpy.ndarray or torch.Tensor, shape (N, G) or (G,)
        represents the points in the source distribution
    b : numpy.ndarray or torch.Tensor, shape (M, G) or (G,)
        represents the points in the target distribution

    Returns
    -------
    masked_l1_mean : torch.Tensor, shape (N, M)
        represents the pairwise masked L1 distance between points in a and b.
    """
    # a: (N, G), b: (M, G)
    if isinstance(a, torch.Tensor) == False:
        a = torch.tensor(a)
    if isinstance(b, torch.Tensor) == False:
        b = torch.tensor(b)
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    a = a.cpu()
    b = b.cpu()

    N, G = a.shape
    M = b.shape[0]

    # Expand a, b to (N, M, G)
    a_expand = a.unsqueeze(1).expand(-1, M, -1)  # (N, M, G)
    b_expand = b.unsqueeze(0).expand(N, -1, -1)  # (N, M, G)

    # Create mask: (N, M, G), True if both a and b are non-zero
    mask = (a_expand != 0) & (b_expand != 0)

    # Compute L1 distance with masking
    abs_diff = torch.abs(a_expand - b_expand) * mask  # (N, M, G)
    valid_counts = mask.sum(dim=2)

    # Sum and normalize
    masked_l1_mean = abs_diff.sum(dim=2) / valid_counts.clamp(min=1)  # Avoid divide-by-zero  # (N, M)

    # Optional: set distance to large value when no valid genes
    masked_l1_mean[valid_counts == 0] = 1e6

    # Check if there are too few valid genes
    # print((mask.sum(dim=2)==0).sum(), (mask.sum(dim=2)==1).sum(), (mask.sum(dim=2)==2).sum(), sep="\t")

    return masked_l1_mean

class GeodesicDistance:
    """Compute the kNN graph distance using scanpy and scipy."""
    def __init__(self, expr, metric="euclidean"):
        self.adata = AnnData(expr)
        scanpy.pp.neighbors(self.adata, use_rep="X", method="umap", metric=metric) # build kNN graph
        self.M = scipy.sparse.csgraph.shortest_path(self.adata.obsp["distances"], directed=False)
        self.knn = scipy.spatial.KDTree(expr.values) # For quick query of point index in expr

    def query(self, x0, x1):
        """Return the precomputed distance between x0 and x1. Only works when x0 and x1 present in the `expr` during initialization."""
        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().numpy()
        if isinstance(x1, torch.Tensor):
            x1 = x1.cpu().numpy()
        x0_idx = self.knn.query(x0, k=1, p=1)[1]
        x1_idx = self.knn.query(x1, k=1, p=1)[1]
        return(torch.tensor(self.M[x0_idx,:][:,x1_idx]))

class CustomMetricOTPlanSampler(OTPlanSampler):
    """Override the OTPlanSampler to use a custom metric for computing the OT plan."""
    def __init__(self, custom_metric, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_metric = custom_metric

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt customized cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        # modified from https://github.com/atong01/conditional-flow-matching/blob/3fd278f9ef2f02e17e107e5769130b6cb44803e2/torchcfm/optimal_transport.py#L63
        
        a, b = ot.unif(x0.shape[0]), ot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)

        # M = torch.cdist(x0, x1) ** 2
        M = self.custom_metric(x0, x1)

        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

# Initialize the SF2M model
sigma = 0.1
sf2m_model = MLP([n_gene, 128, 128, 128, 128, 128, n_gene], time_varying=True).to(device) # Flow matching model
sf2m_score_model = MLP([n_gene, 128, 128, 128, 128, 128, n_gene], time_varying=True).to(device) # Score matching model
# sf2m_model = MLP([n_gene, 64, 64, 64, 64, 64, n_gene], time_varying=True).to(device)
# sf2m_score_model = MLP([n_gene, 64, 64, 64, 64, 64, n_gene], time_varying=True).to(device)
sf2m_optimizer = torch.optim.AdamW(
    list(sf2m_model.parameters()) + list(sf2m_score_model.parameters()), 1e-4
)
SF2M_scheduler = torch.optim.lr_scheduler.StepLR(sf2m_optimizer, step_size=100, gamma=0.5)
SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method="exact")
geo_distance = GeodesicDistance(gene_expression, metric=masked_l1_mean_distance) # kNN + cell similarity
SF2M.ot_sampler = CustomMetricOTPlanSampler(custom_metric=geo_distance.query, method=SF2M.ot_method, reg=2*SF2M.sigma**2) # override the original OTPlanSampler

# For ablation study
if args.no_mask_l1 or args.no_knn:
    if args.no_mask_l1:
        metric = "euclidean"
    else:
        metric = masked_l1_mean_distance
    if args.no_knn:
        SF2M.ot_sampler = CustomMetricOTPlanSampler(custom_metric=metric, method=SF2M.ot_method, reg=2*SF2M.sigma**2)
    else:
        geo_distance = GeodesicDistance(gene_expression, metric=metric)
        SF2M.ot_sampler = CustomMetricOTPlanSampler(custom_metric=geo_distance.query, method=SF2M.ot_method, reg=2*SF2M.sigma**2)

# Data transformation to restrict the output range of Neural ODE as non-negative
class Scaler:
    def transform(self, x):
        x += 1e-6
        return np.where(x<1, np.log(x)+1, x)
    def inverse_transform(self, x):
        x -= 1e-6
        return np.where(x<1, np.exp(x-1), x)
scaler = Scaler()

# Training
pbar = tqdm(range(1000)) # Progress bar
for i in pbar:
    sf2m_optimizer.zero_grad()
    t, xt, ut, eps = get_batch(SF2M, [scaler.transform(i) for i in gene_expression_list[branch_id]], 64, n_snapshot, return_noise=True) # Get a batch of data with batch size 64
    lambda_t = SF2M.compute_lambda(t % 1)
    vt = sf2m_model(torch.cat([xt, t[:, None]], dim=-1))
    st = sf2m_score_model(torch.cat([xt, t[:, None]], dim=-1))
    flow_loss = torch.mean((vt - ut) ** 2)
    score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
    loss = flow_loss + score_loss
    loss.backward()
    sf2m_optimizer.step()
    SF2M_scheduler.step()
    pbar.set_description(f"Loss: {loss.item():.4f}, Flow: {flow_loss.item():.4f}, Score: {score_loss.item():.4f}")
pbar.close()

# Initialize the Neural ODE integrator
node = NeuralODE(torch_wrapper(sf2m_model), solver="dopri5", sensitivity="adjoint", atol=1e-7, rtol=1e-7, atol_adjoint=1e-7, rtol_adjoint=1e-7)

# Generate trajectories
multiplier = 5 # oversampling factor
n_ts = (n_snapshot-1) * multiplier + 1
times = np.linspace(0, n_snapshot-1, n_ts)
trajs = []
for k in range(1): # for over-sampling if > 1
    with torch.no_grad():
        for i in range(n_snapshot):
            tem = np.zeros((n_ts, gene_expression_list[branch_id][i].shape[0], n_gene))
            if i != n_snapshot-1:
                # forward
                traj = node.trajectory(
                    torch.from_numpy(scaler.transform(gene_expression_list[branch_id][i])).float().to(device),
                    t_span=torch.tensor(times[i*multiplier:]).float(),
                ).cpu().numpy()
                tem[i*multiplier:] = traj
            if i != 0:
                # backward
                traj = node.trajectory(
                    torch.from_numpy(scaler.transform(gene_expression_list[branch_id][i])).float().to(device),
                    t_span=torch.tensor(times[:i*multiplier+1][::-1].copy()).float(),
                ).cpu().numpy()[::-1]
                tem[:i*multiplier+1] = traj
            trajs.append(tem)
traj = np.concatenate(trajs, axis=1)
traj = scaler.inverse_transform(traj)

# Calculation of Jacobian matrices, can be treated as GRN but not recommended, comment out if not needed
x_in = torch.from_numpy(gene_expression.values.reshape(-1, n_gene)).to(device).float()
t = []
for i in range(n_snapshot):
    t.append(np.ones(gene_expression_list[branch_id][i].shape[0]) * i)
t = np.concatenate(t, axis=0)
x_in = torch.cat([x_in, torch.from_numpy(t).to(device).float()[:,None]], dim=-1)
x_in.requires_grad = True

jacs = []
for i in range(n_gene):
    jacs.append(torch.autograd.grad(sf2m_model(x_in)[:,i].sum(), x_in)[0][:,:-1].detach().cpu().numpy())
jacs = np.stack(jacs, axis=1)

# Save the results
torch.save(sf2m_model.state_dict(), args.out_dir / 'sf2m_model.pt') # Flow matching model
torch.save(sf2m_score_model.state_dict(), args.out_dir / 'sf2m_score_model.pt') # Score matching model
np.save(args.out_dir / 'traj.npy', traj) # Trajectories
np.save(args.out_dir / 'jacs.npy', jacs) # Jacobian matrices