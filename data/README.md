# Dataset files

This project uses the PeMSD7(M) and PeMS-BAY traffic forecasting datasets.
The full data files are not committed to this repository because they are large and may be downloaded from the link given in the main README.

Expected files:

```text
V_228.csv      # PeMSD7(M) average speed matrix, 228 road segments
W_228.csv      # PeMSD7(M) weighted adjacency matrix
pems-bay.h5    # PeMS-BAY traffic speed data, 325 sensors
W_Bay.csv      # PeMS-BAY weighted adjacency matrix
```

Place the files either in the repository root or in this `data/` directory, then pass their paths through the command-line arguments shown in the main README.


