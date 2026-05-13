# Refactor plan

The current repository preserves the original project scripts in `legacy/`. A future refactor should make the code easier to run and maintain.

Recommended steps:

1. Move reusable model components into `src/models/`:
   - spectral graph convolution
   - graph self-attention
   - STGT transformer block
   - standard transformer baseline
2. Move data loading and preprocessing into `src/data/`.
3. Replace separate scripts with one configurable entry point:
   ```bash
   python train.py --model stgt --dataset pemsd7m --mode autoregressive
   python train.py --model stgt --dataset pems-bay --mode multi-step
   ```
4. Add YAML config files for PeMSD7(M) and PeMS-BAY.
5. Add a small synthetic-data smoke test so the repository can be tested without downloading the full datasets.
6. Add experiment logging and save results in CSV format.
