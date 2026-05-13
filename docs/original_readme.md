# Spatio Temporal Graph Transformer

STGT_Files.zip contains 5 pyhton files (.py), requirement.txt, README.md, V_228.csv, W_228.csv, pems-bay.h5 and W_Bay.csv.
They contain the code, data, dependencies and other information required to replicate the results presented in the main paper.

## Dataset
###PeMSD7(M) Dataset :
* Average Speed data :  V_228.csv
* Adjacency Matrix :  W_228.csv

###PeMS-BAY Dataset: 
* Average Speed data :  pems-bay.h5
* Adjacency Matrix :  W_Bay.csv


## Training the STGT model
### Setting up the virtual environment
To avoid compatibility issues, it is recommended to execute these codes in a virtual environment. Please go through the following steps:

* Ensure that your system has virtualenv installed. If it does not, you can install it by executing **"pip install virtualenv"** in the command prompt.

* Create a new directory using **"mkdir STGT"**. This directory will store all files and dependencies required for training theSTGT model and its variants. Then make this your working directory using **"cd STGT"** command.

* Unpack STGT_Files.zip and copy all its contents into the newly created STGT directory.

* Create a new virtual environment by typing **"virtualenv STGT_env"**. After that activate this environment by executing **"source STGT_env/bin/activate"** in the command prompt.

* Execute **"pip install -r requirements.txt"** to install the dependencies.

* The virtual environment is now ready.

### Reproducing the results in the main paper
To run a .py file execute **"python main.py"** in the command prompt.

Provide the following arguments while running the codes:

* --L : Number of transformer blocks (default=3)

* --H : Number of attention heads per transformer block (default=3)

* --dp : Path to average speed data (default='None')

* --amp : Path to adjacency matrix (default='None')

* --d : PeMS-Bay or PeMSD7(M) (default='PeMS-BAY')

* To reproduce the results of Table 1 standard transformer copy and execute **python standard_transformer.py --L 3 --H 3 --dp 'V_228.csv' --amp 'W_228.csv' --d 'PeMSD7(M)'** on the command prompt.

* To reproduce the results of Table 1 STGT copy and execute **python STGT_transformer.py --L 3 --H 3 --dp 'V_228.csv' --amp 'W_228.csv' --d 'PeMSD7(M)'** on the command prompt.

* To reproduce the results of Table 2 STGT copy and execute **python STGT_transformer_MSP.py --L 5 --H 3 --dp 'V_228.csv' --amp 'W_228.csv' --d 'PeMSD7(M)'** on the command prompt.

* To reproduce the results of Table 3 standard transformer copy and execute **python standard_transformer.py --L 3 --H 3 --dp 'pems-bay.h5' --amp 'W_Bay.csv' --d 'PeMS-Bay'** on the command prompt.

* To reproduce the results of Table 3 STGT copy and execute **python STGT_transformer.py --L 2 --H 3 --dp 'pems-bay.h5' --amp 'W_Bay.csv' --d 'PeMS-Bay'** on the command prompt.

* To reproduce the results of Table 4 STGT copy and execute **python STGT_transformer_MSP.py --L 4 --H 3 --dp 'pems-bay.h5' --amp 'W_Bay.csv' --d 'PeMS-Bay'** on the command prompt.

* To reproduce the results of Table 5 STGT copy and execute **python STGT_transformer.py --L 3 --H 3 --dp 'V_228.csv' --amp 'W_228.csv' --d 'PeMSD7(M)'** on the command prompt. Here, vary H 3, 6, 9, 12, 15. 

* To reproduce the results of Table 6 standard transformer copy and execute **python standard_transformer.py --L 3 --H 3 --dp 'V_228.csv' --amp 'W_228.csv' --d 'PeMSD7(M)'** on the command prompt.

* To reproduce the results of Table 6 STGT copy and execute **python STGT_transformer.py --L 3 --H 3 --dp 'V_228.csv' --amp 'W_228.csv' --d 'PeMSD7(M)'** on the command prompt.

* To reproduce the results of Table 6 STGT-NGT copy and execute **python STGT_NGT_transformer_MSP.py --L 3 --H 3 --dp 'V_228.csv' --amp 'W_228.csv' --d 'PeMSD7(M)'** on the command prompt.

* To reproduce the results of Table 6 STGT-NGP copy and execute **python STGT_NGP_transformer_MSP.py --L 3 --H 3 --dp 'V_228.csv' --amp 'W_228.csv' --d 'PeMSD7(M)'** on the command prompt.


### Deactivating the virtual environment
You may deactivate the virtual environment by executing **"deactivate"**.
