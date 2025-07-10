HOW TO RUN THIS PROJECT:

Requirements:
GPU, Git Bash, Python 3.10(not 3.13), Visual Studio Build tools, CUDA Toolkit

Steps:
1. Navigate to this folder in vs code
2. Open git bash
3. run:
py -3.10 -m venv venv
.\venv\Scripts\activate          or 
source venv/Scripts/activate (for git bash its the source command)

(now it should (venv) above your username part in bash)

4. Check what version of GPU you have by typing nvidia-smi
accordingly, install the CORRECT pytorch version
# for CUDA 12.1
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
or else check out pytorch website: https://pytorch.org/get-started/previous-versions/

5. install pytorch geometric from this link according to your cuda etc(might hav eto just manually download the file if git bash command below doesnt work): https://pytorch-geometric.com/whl/

# CUDA 12.1 + torch 2.1.0
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric==2.5.3

6. other packages install:
pip install networkx trimesh pyglet tqdm numpy scipy scikit-learn matplotlib pyyaml

7. Run the modelnet10_to_graph.py (and remember that it downloads the modelnet10 dataset AND turns it into pkl files) using git bash:
python modelnet10_to_graph.py

8. The initial parameters of the yaml file should be:
# config/modelnet10_graphs.yaml
data_root: data/graph_data
batch_size: 16
hidden_dim: 64
num_classes: 10
lr: 0.0008
epochs: 50
num_layers: 3
curvature: 1.0
weight_decay: 1e-4
step_size: 20
gamma: 0.5

You can play around with these to get better accuracy

9. Finally train the model- will take a few hours btw dont freak out about it:
python train.py --cfg config/modelnet10_graphs.yaml

10. now about the test dataset, and performance per class:
python test.py

11. clean and deactivate your environment:
rm -rf __pycache__/
rm -rf data/graph_data/*
rm -rf checkpoints/*

deactivate






