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

Pretrained model preparation: download the pretrained models to ./cache
```shell
python code/model_cache.py
```

Run the baselines: roughly 40GB GPU memory is required.
```shell
# Change the ${WORKDIR} and ${DATADIR} to the corresponding path
vim code/exp_with_args.sh

# Run the experiments
# The multi-task scripts takes around 5 days on a single V100 GPU(32GB)
# CodeT5
python code/run_exp.py --model_tag codet5_base --task multi_task --sub_task none --gpu xxx --gas xxx
# Use --gas to specify gradient accumulate steps: e.g. --gas 2, --gas 4 ...
# Use --gpu to specify gpu indices: e.g. --gas 0,1,2,3 for using the first 4 GPUs to run the experiment
python code/run_exp.py --model_tag codet5_base --task refine --sub_task medium --gpu xxx --gas xxx

# T5
python code/run_exp.py --model_tag t5_base --task multi_task --sub_task none --gpu xxx --gas xxx
python code/run_exp.py --model_tag t5_base --task refine --sub_task medium --gpu xxx --gas xxx

# CoTexT
python code/run_exp.py --model_tag cotext --task multi_task --sub_task none --gpu xxx --gas xxx
python code/run_exp.py --model_tag cotext --task refine --sub_task medium --gpu xxx --gas xxx
```