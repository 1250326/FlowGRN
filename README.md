# FlowGRN: Scalable GRN Inference with Dropout-Robust Neural ODEs

This repository contains the official implementation of "FlowGRN: Scalable and Dropout-Robust Gene Regulatory Network Inference via Flow Matching-Based Trajectory Reconstruction".

# Setup
## Environment

Please note that the following is only a record of our environment, and we believe our code can adapt to a wide range of versions.

- Python: 3.9.18
- anndata: 0.10.5
- numpy: 1.23.5
- scanpy: 1.9.6
- scipy: 1.12.0
- tqdm: 4.66.1
- pandas: 2.2.3
- pytorch: 2.2.1
- pytorch-cuda: 11.8
- torchdyn: 1.0.6
- torchcfm: 1.0.7
- sklearn: 1.3.1
- pot: 0.9.3
- decoupler-py: 1.8.0
- networkx: 3.1

A `requirements.txt` is provided and users can install the environment via:
```python
pip install -r requirements.txt
```

# Data source
This research does not generate any new data.
The BEELINE datasets used in this research are publicly available at [https://zenodo.org/records/3701939](https://zenodo.org/records/3701939).

The ground truth networks for experimental datasets used in this research can be downloaded using python:
```python
import decoupler as dc

human_collectri_df = dc.get_collectri(organism='human', split_complexes=False)
human_dorothea_df = dc.get_dorothea(organism='human', split_complexes=False, levels=tuple("ABCD"))

mouse_collectri_df = dc.get_collectri(organism='mouse', split_complexes=False)
mouse_dorothea_df = dc.get_dorothea(organism='mouse', split_complexes=False, levels=tuple("ABCD"))
```

# Training
The training process contains two steps: trajectory reconstruction and network inference.

Since the former step is GPU-intensive and the latter step is CPU-intensive, we would recommend users to run the two steps separately on different platforms.
Our code is also designed to be run on different platforms and maximize its parallelization.

For simulated datasets from BEELINE, it is more convenient to use the script `get_traj_sim.py` to reconstruct trajectories and use `run_sim.py` to infer networks.

For experimental datasets, users can use `get_traj_real_dropout.py` to reconstruct trajectories and `run_real_scatter.py` to infer networks.

The main difference between two pipelines is that for the simulated datasets, we don't need to handle dropout as the simulated data is dropout-free, while for the experimental datasets, we need to handle dropout in the trajectory reconstruction step.
Also, the experimental datasets are much higher dimensional, so we scatter the training of dynGENIE3 to multiple computing clusters.

## Data preparation

All datasets should equipped with a `.csv` file describing the pseudotime of each cell.
Example file `PseudoTime.csv` is provided in the `sample_inputs/` folder, extracted from the dyn-CY-2000 dataset.

For simulated datasets, the gene expression file is a `.csv` file with genes as rows and cells as columns, with the first column being the gene names and the first row being the cell names.
Example file `ExpressionData.csv` is provided in the `sample_inputs/` folder, extracted from the dyn-CY-2000 dataset.

For experimental datasets, the gene expression file may either be:
- a `.csv` file with the same format as the simulated datasets, or
- a pickle file containing a list of list of np.ndarrays, where each ndarray is a gene expression matrix (cell x gene) for a time snapshot for each branch.
  
Example file `gene_expression_list.pkl` is provided in the `sample_inputs/` folder, generated from the mHSC-L dataset with "TFs + 500 genes".
It has the following structure:
```python
[
    [               # Branch 0 with 3 time snapshots
        np.array(), # 357 cells x 624 genes
        np.array(), # 228 cells x 624 genes
        np.array(), # 262 cells x 624 genes
    ]
]
```

## Trajectory reconstruction
Trajectories are reconstructed by the following command:

For simulated datasets:
```bash
python get_traj_sim.py 
    --expr-fn sample_inputs/ExpressionData.csv
    --pseudoT-fn sample_inputs/PseudoTime.csv
    --out-dir <output_directory>
    --seed <random_seed>
    --branch-id <branch_id>
```
Note that the `branch-id` is corresponding to the branch index in the pseudotime file, starting from 0.
Users should iterate over all branches in the pseudotime file to reconstruct trajectories for all branches, and then concatenate the results into a single trajectory file.

For experimental datasets:
```bash
# For csv files, time snapshots are evenly spaced
python get_traj_real_dropout.py
    --expr-fn sample_inputs/ExpressionData.csv
    --pseudoT-fn sample_inputs/PseudoTime.csv
    --out-dir <output_directory>
    --seed <random_seed>
    --branch-id <branch_id>
    --n-snapshot <total_number_of_time_snapshots>

# For pickle files, time snapshots are pre-defined
python get_traj_real_dropout.py
    --precomputed-batch-fn sample_inputs/gene_expression_list.pkl
    --out-dir <output_directory>
    --seed <random_seed>
    --branch-id <branch_id>
```

To reproduce the results of the ablation on knn graph and the new cell similarity measure, or you simply don't want to use them, you can add the following options `--no-knn` and `--no-mask-l1` respectively.

The outputs will be a new directory containing the following files:
- `sf2m_model.pt`: the trained CFM model (flow matching)
- `sf2m_score_model.csv`: the trained CFM model (score matching)
- `traj.npy`: the reconstructed trajectories with shape (n_time, n_cell, n_gene). We reconstructed trajectories for each cell.
- `jacs.npy`: the Jacobian matrices of the flow matching model evaluated at each cell with shape (n_cell, n_gene, n_gene).
- `gene_list.npy`: an array of the gene names in the order of the input expression data, only for simulated datasets.

## Network inference
Before running the network inference, users should concatenate the trajectories of all branches into a single file, say `traj.npy`, with shape (n_time, n_cell, n_gene).

The network inference can be run by the following command:
```bash
# For simulated datasets:
python run_sim.py traj.npy <output_filename>

# For experimental datasets:
python run_real_scatter.py traj.npy <output_directory> <gene_id>
```

The `gene_id` is the index of the gene starting from 0, corresponding the the index in the `traj.npy` file.

For simulated datasets, the output will be a `.npz` file containing the following keys:
- `VIM`: inferred network
- `alphas`: estimated alphas
- `prediction_score`: prediction scores
- `stability_score`: stability scores

For experimental datasets, the output will be a new directory containing the intermediate results of the dynGENIE3 algorithm for each gene, each file is named as `<gene_id>.npz` and contains the following keys:
- `vi`: `<gene_id>`-th column of the inferred network
- `alphas`: estimated alphas
- `prediction_score_i`: prediction scores
- `stability_score_i`: stability scores

The definition of the alphas, prediction scores, and stability scores can be found in the dynGENIE3 paper.

Users may concatenate the inferred networks of all genes into a single file for the experimental datasets with the following command:
```python
import numpy as np

VIM = np.zeros((n_genes, n_genes))  # Initialize an empty matrix for the inferred network
for gene_id in range(n_genes):
    data = np.load(f'<output_directory>/{gene_id}.npz')
    VIM[gene_id, :] = data['vi']  # Fill the row with the inferred network for the gene
np.save('<output_filename>', VIM)  # Save the inferred network
```

# Evaluation
We use AUPRC and EPR to evaluate the inferred networks, and their definitions can be found in our paper.

AUPRC is calculated by the function `sklearn.metrics.average_precision_score`.

EPR is calculated by the function:
```python
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
```
<!-- # Citation
If you use this code in your research, please cite our paper: -->