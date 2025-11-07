# Flow-Learning

Personal implementation of flow matching for imitation learning. Code is mostly from scratch, with references to openpi and 3d diffusion for details.

Implementation is based on the following papers:

- https://www.physicalintelligence.company/download/pi0.pdf for flow matching training process
- https://arxiv.org/abs/2403.03954 for model architecture

Code is quite messy and needs cleanup still, but does function.

Makes use of the robosuite simulation environment and collecting data, and the robomimic library for converting that data to batch formats, and for data loaders for torch. 

## Setup: 
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
```

## Data
Collect data: `python3 data_collect_robosuite.py`
Process data for training: `./convert_hdf5.sh <dataset_path> <output_name>`