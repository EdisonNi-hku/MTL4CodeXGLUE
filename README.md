# MTL4CodeXGLUE
Multi-task Learning for CodeXGLUE Benchmark

The experiment scripts are adapted from: https://github.com/salesforce/CodeT5. The original scripts does not support resuming training & multi-gpu fine-tuning.
Environment preparation
```shell
git clone https://github.com/EdisonNi-hku/MTL4CodeXGLUE
conda create -n codemtl
source activate codemtl
cd MTL4CodeXGLUE
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r code/requirements.txt
```

Data preparation
```shell
pip install gsutil
cd MTL4CodeXGLUE
gsutil -m cp -r "gs://sfr-codet5-data-research/data" .
```

Extra data for auxiliary tasks:
```shell
# Please first download dataflow.zip from: https://drive.google.com/u/0/uc?id=1So0qrXTM2EUX2LiRrGT_YNuB2hh1b3s-&export=download
unzip dataflow.zip -d data
cp -r data/dataflow data/identifier
```

Pretrained model preparation: download the pretrained models to ./cache
```shell
python code/model_cache.py
```

Run the baselines: roughly 40GB GPU memory is required.
```shell
# Change the ${WORKDIR} and ${DATADIR} to the corresponding path
vim code/exp_with_args.sh

# Run the experiments
# CodeT5
python code/run_exp.py --model_tag codet5_base --task multi_task --sub_task none --gpu 0 --gas 2 --prefix 0
# Cmd for multi-task experiment
# Use --gas to specify gradient accumulate steps: e.g. --gas 2, --gas 4 ...
# Use --gpu to specify gpu indices: e.g. --gpu 0,1,2,3 for using the first 4 GPUs to run the experiment
# Use --prefix to specify what prefix to use, 0(default value) for no prefix, 1 for only source prefix, 2 for target + source prefix, 3 for only target prefix
python code/run_exp.py --model_tag codet5_base --task multi_auxiliary --sub_task none --gpu 0 --gas 2 --aux_percentage 10 --prefix 0 --aux_type 0
# Cmd for multi-task + auxiliary tasks
# Use --aux_percentage (default value: 10) to specify how many data will be used in auxiliary tasks.
# Use --aux_type (default value: 0) to specify which auxiliary task to use, 0 for both, 1 for dataflow prediction, 2 for identifier denoising

# T5
python code/run_exp.py --model_tag t5_base --task multi_task --sub_task none --gpu 0 --gas 2 --prefix 0
python code/run_exp.py --model_tag t5_base --task multi_auxiliary --sub_task none --gpu 0 --gas 2 --aux_percentage 10 --prefix 0 --aux_type 0

# CoTexT
python code/run_exp.py --model_tag cotext --task multi_task --sub_task none --gpu 0 --gas 2 --prefix 0
python code/run_exp.py --model_tag cotext --task multi_auxiliary --sub_task none --gpu 0 --gas 2 --aux_percentage 10 --prefix 0 --aux_type 0
```
