# Flow-Learning

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