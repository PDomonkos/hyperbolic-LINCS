# Hyperbolic Nature of Differential Expression Signatures

Analysis of LINCS DEG signatures for the paper "Hyperbolic Nature of Differential Expression Signatures"

Contact Information: pogany@mit.bme.hu

## Data accession

In our study, we have utilized the L1000FWD [[1]](#1) and SigCom LINCS [[2]](#2) datasets provided by the NIH Library of Integrated Network-Based Cellular Signatures (LINCS) Program.

To replicate the results download the following data from the [L1000FWD webpage](https://maayanlab.cloud/l1000fwd/download_page) and the [SigCom LINCS portal](https://maayanlab.cloud/sigcom-lincs/#/Download) to `data/L1000FWD` and `data/SigComLINCS` respectively:
- L1000FWD:
    - Drug-Induced Gene Expression Signatures and Adjacency Matrices:
        - CD_signatures_full_42809x22268.gctx
        - D_signatures_LM_42809x978.gctx
    - Graphs: 
        - All cell lines
- SigCom LINCS
    - L1000 Characteristic Direction Coefficient Tables (Level 5):
        - LINCS L1000 Chemical Perturbations (2021)
    - LINCS Small Molecules Metadata:
        - LINCS Small Molecules Metadata

## Requirements

To run the notebooks, we utilized Python 3.12.4 and the following packages:
- `pandas` (version 2.2.2)
- `numpy` (version 1.26.4)
- `cmapPy` (version 4.0.1)
- `networkx` (version 3.3)
- `tqdm` (version 4.66.4)
- `matplotlib` (version 3.9.1)
- `seaborn` (version 0.13.2)
- `statsmodels` (version 0.14.2)
- `scipy` (version 1.14.0)
- `scikit-learn` (version 1.5.1)
- `torch` (version 2.3.0)
- `geoopt` (version 0.5.1)
- `umap-learn` (version 0.5.6)
- `poincare_maps`: https://github.com/facebookresearch/PoincareMaps/tree/main [[3]](#3) 

## Contents

- `data.py`: Load the L1000FWD and SigCom LINCS signatures and metadata.
- `dimred.py`: Wrapper functions for the dimensionality reduction techniques.
- `eval.py`: Evaluate local and global structure preservation.
- `network.py`: Synthetic network generation and scale-free soft thresholding analysis.
- `hyperparameters.ipynb`: Hyperparameter search for the various dimensionality reduction methods.
- `comparison.ipynb`: Compare dimensionality reduction methods across different dimensions based on local and global structure preservation.
- `2D_visualization.ipynb`: Visualize the 2D signature embeddings.
- `scale_free_nature.ipynb`: Analysis on the scale-free nature of the differential expression signatures.

## References
<a id="1">[1]</a> 
Wang, Z., Lachmann, A., Keenan, A. B., & Ma’Ayan, A. (2018). 
L1000FWD: fireworks visualization of drug-induced transcriptomic signatures. 
Bioinformatics, 34(12), 2150-2152.

<a id="2">[2]</a> 
Evangelista, J. E., Clarke, D. J., Xie, Z., Lachmann, A., Jeon, M., Chen, K., ... & Ma’ayan, A. (2022). 
SigCom LINCS: data and metadata search engine for a million gene expression signatures. 
Nucleic acids research, 50(W1), W697-W709.

<a id="3">[3]</a> 
Klimovskaia, A., Lopez-Paz, D., Bottou, L., & Nickel, M. (2020). 
Poincaré maps for analyzing complex hierarchies in single-cell data. 
Nature communications, 11(1), 2966.

## Citation
```
TBA
``` 

Preprint DOI: [10.36227/techrxiv.24630747.v1](https://www.techrxiv.org/doi/full/10.36227/techrxiv.24630747.v1)
