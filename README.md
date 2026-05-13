# Spatio-Temporal Graph Transformer for Traffic State Prediction

This repository contains the code for an **unpublished research project** on Spatio-Temporal Graph Transformers (STGT) for traffic state prediction.

STGT incorporates spectral graph convolution into transformer blocks to improve traffic forecasting on graph-structured road-network data. The project explores two insertion points for spectral graph convolution:

1. query/key/value projection inside self-attention, and
2. the position-wise transformation after scaled dot-product self-attention.

The original project scripts are preserved in `legacy/`. A cleaner CPU/GPU package refactor is planned under `docs/refactor_plan.md`.

## Project summary

Traffic speed forecasting is a spatio-temporal graph learning problem: each road segment is a graph node, the road network defines the adjacency matrix, and each node has a time series of speed readings. STGT combines:

- temporal encoding for the lookback window,
- multi-head transformer attention for long-range temporal modeling, and
- spectral graph convolution for road-network spatial structure.

The project was evaluated on two Caltrans PeMS traffic datasets:

- **PeMSD7(M):** 228 road segments
- **PeMS-BAY:** 325 sensors

Both datasets use 5-minute aggregation intervals, and the experiments evaluate 15/30/45-minute prediction horizons.

## Repository structure

```text
.
├── legacy/                 # Original STGT project scripts
├── scripts/                # Convenience run scripts
├── results/                # Tables extracted from project results
├── data/                   # Dataset placeholder; data files are not committed
├── docs/                   # Original README and refactor notes
├── requirements.txt
└── README.md
```

## Dataset

The full dataset files are not committed to this repository. Place the following files in `data/`:

```text
data/V_228.csv      # PeMSD7(M) speed data
data/W_228.csv      # PeMSD7(M) adjacency matrix
data/pems-bay.h5    # PeMS-BAY speed data
data/W_Bay.csv      # PeMS-BAY adjacency matrix
```

See `data/README.md` for details.

## Environment

The original code was developed with TensorFlow/Keras. A virtual environment is recommended.

```bash
python -m venv stgt_env
source stgt_env/bin/activate
pip install -r requirements.txt
```

The pinned dependencies reflect the original project environment and may require an older Python/TensorFlow stack.

## Running the original scripts

PeMSD7(M), autoregressive STGT:

```bash
bash scripts/run_pemsd7m_stgt_ar.sh
```

PeMSD7(M), multi-step STGT:

```bash
bash scripts/run_pemsd7m_stgt_multistep.sh
```

PeMS-BAY, autoregressive STGT:

```bash
bash scripts/run_pemsbay_stgt_ar.sh
```

PeMS-BAY, multi-step STGT:

```bash
bash scripts/run_pemsbay_stgt_multistep.sh
```

Equivalent direct commands are documented in `docs/original_readme.md`.

## Results

The full result table is available in `results/stgt_results.csv`.

### PeMSD7(M), multi-step prediction

| Model | MAE 15/30/45 min | MAPE 15/30/45 min | RMSE 15/30/45 min |
|---|---|---|---|
| Graph WaveNet | 2.14 / 2.80 / 3.19 | 4.93 / 6.89 / 8.04 | 4.01 / 5.48 / 6.25 |
| STTN | 2.14 / 2.70 / 3.03 | 5.05 / 6.68 / 7.61 | 4.04 / 5.37 / 6.05 |
| STGT | 2.298 / 2.299 / 2.307 | 5.234 / 5.272 / 5.274 | 3.754 / 3.763 / 3.767 |

### PeMS-BAY, multi-step prediction

| Model | MAE 15/30/45 min | MAPE 15/30/45 min | RMSE 15/30/45 min |
|---|---|---|---|
| Graph WaveNet | 1.30 / 1.63 / 1.95 | 2.74 / 3.70 / 4.52 | 2.73 / 3.67 / 4.63 |
| STTN | 1.36 / 1.67 / 1.95 | 2.89 / 3.78 / 4.58 | 2.87 / 3.79 / 4.50 |
| STGT | 1.560 / 1.562 / 1.558 | 3.155 / 3.156 / 3.160 | 2.751 / 2.815 / 2.752 |

The strongest results are in long-horizon multi-step forecasting, where STGT substantially reduces 30/45-minute RMSE compared with Graph WaveNet and STTN on both datasets.

## Notes

This repository is intended as a public code archive and starting point for future cleanup. The project was not peer-reviewed or formally published, so please cite the repository rather than treating it as a publication.
