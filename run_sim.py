import numpy as np

import sys
from dynGENIE3 import dynGENIE3

# Read filename and output directory from command line arguments
fn, outdir = sys.argv[1:]

# Load the trajectory
traj = np.load(fn)

n_time, n_sim, n_gene = traj.shape
timestep = [np.arange(n_time) for i in range(n_sim)]
TS_data = [traj.transpose(1,0,2)[i] for i in range(n_sim)]

# Run dynGENIE3
VIM, alphas, prediction_score, stability_score, _ = dynGENIE3(TS_data, timestep, compute_quality_scores=True, nthreads=1, ntrees=1000)

# Save the results
np.savez(outdir, VIM=VIM, alphas=alphas, prediction_score=prediction_score, stability_score=stability_score)