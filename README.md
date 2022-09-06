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

CodeXGLUE Data preparation
```shell
pip install gsutil
cd MTL4CodeXGLUE
gsutil -m cp -r "gs://sfr-codet5-data-research/data" .
```

Download the 5% data for MTL from: https://drive.google.com/file/d/1rzxvnBwTuqkZQ2yhWnfQGVOkOXjabBKK/view?usp=sharing
```shell
unzip data_5.zip -d data_5
```

Pretrained model preparation: download the pretrained models to ./cache
```shell
python code/model_cache.py
```

Experiments:
```shell
# Change the ${WORKDIR} and ${DATADIR} to the corresponding path
vim code/exp_with_args.sh

# Commands examples
python code/run_exp.py --model_tag codet5_base --task summarize --sub_task python --gpu 0 --gas 2
# Single task fine-tuning on CodeT5
# Use --model_tag to specify pre-trained model to use: e.g. codet5_base, cotext, t5_base...
# Use --gas to specify gradient accumulate steps: e.g. --gas 2, --gas 4 ...
# Use --gpu to specify gpu indices: e.g. --gpu 0,1,2,3 for using the first 4 GPUs to run the experiment

python code/run_exp.py --model_tag codet5_base --task multi_task --sub_task none --gpu 0 --gas 2 --data_dir data_5 --max_step 50000 --save_step 1000 --free
# MTL without auxiliary task
# Use --data_dir to specify the data directory
# Use --max_step and --save_step to control training schedule(here we use smaller steps since we only use 5% of data)
# --free flag denotes that the step parameters are free to set.

python code/run_exp.py --model_tag codet5_small --task multi_auxiliary --sub_task none --gpu 0 --data_dir data_5 --max_step 50000 --save_step 1000 --free --aux_percentage 10 --aux_type 0123
# Cmd for multi-task + auxiliary tasks
# Use --aux_percentage (default value: 10) to specify how many data will be used in auxiliary tasks.
# Use --aux_type (default value: 0) to specify which auxiliary task to use, 0 for data flow generation, 1 for identifier denoising, 2 for summarization+SRL, 3 for translation+cloze
```
