# PowerGraph-node
# Benchmarking GNN datasets for PowerGrids- node level regression tasks

With the InMemoryDatasets Class, we generate the GNN datasets for the UK, IEEE24, IEEE39, IEEE118 power grids. We use **InMemoryDataset** class of Pytorch Geometric.

## Installation

**Requirements**

- CPU or NVIDIA GPU, Linux, Python 3.7
- PyTorch >= 1.5.0, other packages

Load every additional packages:

```
pip install -r requirements.txt
```

## Prerequisites and data structure

To reproduce the results presented in the paper, download the following compressed data from [here](https://figshare.com/articles/dataset/PowerGraph/22820534?file=46619152) (~1.08GB, when uncompressed):

```bash
wget -O data.tar.gz "https://figshare.com/ndownloader/files/46619152"
tar -xf data.tar.gz
```

Each dataset folder contains the following files:

- `edge_attr.mat`: edge feature matrix for the power flow problem
- `edge_attr_opf.mat`: edge feature matrix for the optimal power flow problem
- `edge_index.mat`: branch list for the power flow problem
- `edge_index_opf.mat`: branch list for the optimal power flow problem
- `X.mat`: node feature matrix for the power flow problem
- `Xopf.mat`: node feature matrix for the optimal power flow problem
- `Y_polar.mat`: node output matrix for the power flow problem
- `Y_polar_opf.mat`: node output matrix for the optimal power flow problem

## Dataset description

| Dataset    |     Name     | Description                    |
| ---------- | :----------: | ------------------------------ |
| IEEE-24    |   `ieee24`   | IEEE-24 (Powergrid dataset)    |
| IEEE-39    |   `ieee39`   | IEEE-39 (Powergrid dataset)    |
| IEEE-118   |  `ieee118`   | IEEE-118 (Powergrid dataset)   |
| UK         |     `uk`     | UK (Powergrid dataset)         |

Graph dataset that models the power flow and optimal power flow problems in the power grid. To generate a comprehensive dataset for different power grids, we use MATPOWER. This model simulates how failures propagate in the IEEE24, IEEE39, IEEE118 and UK power grids. The output of the model is the final demand not being served (DNS). Our dataset consists of a large set of power grid states, representing the operating conditions before an outage, and is linked to an initial triggering outage (one or more failed elements). Each power grid state is represented as a graph, with a graph-level label assigned based on the results of the physics-based model. The dataset is designed for various graph-level tasks, such as multi-class classification, binary classification, and regression. Bus and branches are the elements of a power grid, buses include loads and generators which represent the nodes of the graph, while branches include transmission lines and transformers which represent the edges of the graph. We provide three features per node: net active power, net apparent power and voltage magnitude. While the features per edge are four: active power flow, reactive power flow, line reactance and line rating.

## GNN Benchmarking

To test the datasets with different GNN architectures: GCN, GINe, GAT and Transformer, run:

```
python code/train_gnn.py
```

We have the main arguments to control namely

**--model_name**: transformer / gin / gat / gcn

**--datatype**: node / nodeopf (for power flow and optimal power flow problems, respectively)

**--dataset_name**: uk / ieee24 / ieee39 / ieee118



Make sure you have the dataset as per format. Models will be saved as per format (make sure you have the model folder)

```
.
├── code
├── dataset
│ ├── processed
│ ├── raw
| | ├── \*.mat
├──model
| ├──ieee24
| ├──ieee39
| ├──uk
| ├──ieee118
```

Remove the for loop in train_gnn.py if running for a specific **--hidden_dim** and **num_layers**.

The models will be saved in **model** directory


## License

This work is licensed under a CC BY 4.0 license.
