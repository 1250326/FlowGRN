import numpy as np
from pathlib import Path

import sys
from dynGENIE3 import estimate_degradation_rates, dynGENIE3_single

# Read filename, output directory, and gene ID from command line arguments
fn, outdir, gene_id = sys.argv[1:]
outdir = Path(outdir)
if not outdir.exists(): # Create the output directory if it does not exist
    outdir.mkdir(parents=True, exist_ok=True)

gene_id = int(gene_id)
traj = np.load(fn) # Load the trajectory

n_time, n_sim, n_gene = traj.shape
timestep = [np.arange(n_time) for i in range(n_sim)]
TS_data = [traj.transpose(1,0,2)[i] for i in range(n_sim)]

# from dynGENIE3: dynGENIE3(TS_data, timestep, compute_quality_scores=True, nthreads=1, ntrees=1000)
ngenes = TS_data[0].shape[1]
alphas = estimate_degradation_rates(TS_data,timestep)
input_idx = list(range(ngenes))
prediction_score = np.zeros(ngenes)
stability_score = np.zeros(ngenes)
# Run single random forest for the specified gene
(vi,prediction_score_i,stability_score_i,treeEstimator) = dynGENIE3_single(TS_data,timestep,None,gene_id,alphas[gene_id],input_idx,"RF","sqrt",1000,True,False)

# Save the results
np.savez(outdir / f"{gene_id}.npz", vi=vi, alphas=alphas, prediction_score_i=prediction_score_i, stability_score_i=stability_score_i)