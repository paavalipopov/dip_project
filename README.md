# Requirements

```
conda create -n dip python=3.11
conda activate dip

conda install pytorch::pytorch torchvision torchaudio -c pytorch
OR
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia

pip install -r requirements.txt
```

# Example
`PYTHONPATH=. python scripts/run_experiments.py mode=exp model=baseline_cnn dataset=class_2`

# To add new datasets:
- add conf file to `src/conf/dataset` (see `src/conf/dataset/class_2.yaml` for example)
- add dataset loaders to `src/datasets` (see `src/datasets/class_2.py` for example)
    - you must define `load_data(cfg: DictConfig)` there

# To add new model: 
- add conf file to `src/conf/model` (see `src/conf/model/baseline_cnn.yaml` for example)
- add model module to `src/models` (see `src/models/baseline_cnn.py` for example)