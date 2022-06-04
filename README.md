# MTL4CodeXGLUE
Multi-task Learning for CodeXGLUE Benchmark

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

Run the baselines: roughly 40GB GPU memory is required(4 * GPUs with 10GB memory is enough).
```shell
# Change the ${WORKDIR} and ${DATADIR} to the corresponding path
vim code/exp_with_args.sh

# Run the experiments
# CodeT5
python code/run_exp.py --model_tag codet5_base --task summarize --sub_task python
python code/run_exp.py --model_tag codet5_base --task summarize --sub_task java
python code/run_exp.py --model_tag codet5_base --task summarize --sub_task php
python code/run_exp.py --model_tag codet5_base --task summarize --sub_task ruby
python code/run_exp.py --model_tag codet5_base --task summarize --sub_task javascript
python code/run_exp.py --model_tag codet5_base --task summarize --sub_task go

python code/run_exp.py --model_tag codet5_base --task concode --sub_task none

python code/run_exp.py --model_tag codet5_base --task translate --sub_task java-cs
python code/run_exp.py --model_tag codet5_base --task translate --sub_task cs-java

python code/run_exp.py --model_tag codet5_base --task refine --sub_task small
python code/run_exp.py --model_tag codet5_base --task refine --sub_task medium

python code/run_exp.py --model_tag codet5_base --task defect --sub_task none

python code/run_exp.py --model_tag codet5_base --task clone --sub_task none

python code/run_exp.py --model_tag codet5_base --task multi_task --sub_task none

# T5
python code/run_exp.py --model_tag t5_base --task summarize --sub_task python
python code/run_exp.py --model_tag t5_base --task summarize --sub_task java
python code/run_exp.py --model_tag t5_base --task summarize --sub_task php
python code/run_exp.py --model_tag t5_base --task summarize --sub_task ruby
python code/run_exp.py --model_tag t5_base --task summarize --sub_task javascript
python code/run_exp.py --model_tag t5_base --task summarize --sub_task go

python code/run_exp.py --model_tag t5_base --task concode --sub_task none

python code/run_exp.py --model_tag t5_base --task translate --sub_task java-cs
python code/run_exp.py --model_tag t5_base --task translate --sub_task cs-java

python code/run_exp.py --model_tag t5_base --task refine --sub_task small
python code/run_exp.py --model_tag t5_base --task refine --sub_task medium

python code/run_exp.py --model_tag t5_base --task defect --sub_task none

python code/run_exp.py --model_tag t5_base --task clone --sub_task none

python code/run_exp.py --model_tag t5_base --task multi_task --sub_task none

# CoTexT
python code/run_exp.py --model_tag cotext --task summarize --sub_task python
python code/run_exp.py --model_tag cotext --task summarize --sub_task java
python code/run_exp.py --model_tag cotext --task summarize --sub_task php
python code/run_exp.py --model_tag cotext --task summarize --sub_task ruby
python code/run_exp.py --model_tag cotext --task summarize --sub_task javascript
python code/run_exp.py --model_tag cotext --task summarize --sub_task go

python code/run_exp.py --model_tag cotext --task concode --sub_task none

python code/run_exp.py --model_tag cotext --task translate --sub_task java-cs
python code/run_exp.py --model_tag cotext --task translate --sub_task cs-java

python code/run_exp.py --model_tag cotext --task refine --sub_task small
python code/run_exp.py --model_tag cotext --task refine --sub_task medium

python code/run_exp.py --model_tag cotext --task defect --sub_task none

python code/run_exp.py --model_tag cotext --task clone --sub_task none

python code/run_exp.py --model_tag cotext --task multi_task --sub_task none
```