import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
from torchdyn.core import NeuralODE
import torchcfm
from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from torchcfm.utils import torch_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--expr-fn', type=str, help="Path of the ExpressionData.csv")
parser.add_argument('--pseudoT-fn', type=str, help="Path of the Pseudotime.csv")
parser.add_argument('--out-dir', type=str, help="Output directory to save the results")
parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
parser.add_argument('--branch-id', type=int, default=0, help="Branch ID to process (0-indexed)")
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

# Load gene expression data and pseudotime
gene_expression = pd.read_csv(args.expr_fn, index_col=0, header=0)
gene_expression = gene_expression.T
gene_list = gene_expression.columns.tolist()

# Load pseudotime data
time_df = pd.read_csv(args.pseudoT_fn, index_col=0, header=0, na_values='NA')
pseudotime = time_df.fillna(0).values.sum(1)
n_cell, n_gene = gene_expression.shape

# Batch gene expression data by branch and time
n_snapshot = 10
n_branch = time_df.shape[1]
assert branch_id < n_branch
gene_expression_list = []
time_split = np.linspace(0, pseudotime.max(), n_snapshot+1)
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

# Initialize the SF2M model
sigma = 0.1
# sf2m_model = MLP([n_gene, 128, 128, 128, 128, 128, n_gene], time_varying=True).to(device)
# sf2m_score_model = MLP([n_gene, 128, 128, 128, 128, 128, n_gene], time_varying=True).to(device)
sf2m_model = MLP([n_gene, 64, 64, 64, 64, 64, n_gene], time_varying=True).to(device) # Flow matching model
sf2m_score_model = MLP([n_gene, 64, 64, 64, 64, 64, n_gene], time_varying=True).to(device) # Score matching model
sf2m_optimizer = torch.optim.AdamW(
    list(sf2m_model.parameters()) + list(sf2m_score_model.parameters()), 1e-4
)
SF2M_scheduler = torch.optim.lr_scheduler.StepLR(sf2m_optimizer, step_size=100, gamma=0.5)
SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method="exact")

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
x_in = torch.cat([x_in, torch.from_numpy(np.repeat(np.arange(n_snapshot), n_cell//n_snapshot)).to(device).float()[:,None]], dim=-1)
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
np.save(args.out_dir / 'gene_list.npy', gene_list) # Ordering of genes