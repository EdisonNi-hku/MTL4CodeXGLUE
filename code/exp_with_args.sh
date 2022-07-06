WORKDIR="/cluster/work/sachan/leonhard/jingwei/ni2/MTL4CodeXGLUE/code"
DATADIR="/cluster/work/sachan/leonhard/jingwei/ni2/MTL4CodeXGLUE/data"
export PYTHONPATH=$WORKDIR

TASK=${1}
SUB_TASK=${2}
MODEL_TAG=${3}
GPU=${4}
DATA_NUM=${5}
BS=${6}
LR=${7}
SRC_LEN=${8}
TRG_LEN=${9}
PATIENCE=${10}
EPOCH=${11}
WARMUP=${12}
MODEL_DIR=${13}
SUMMARY_DIR=${14}
RES_FN=${15}
LOAD_PATH=${16}
GRADIENT_STEP=${17}
EVAL_BS=${18}
AUX_PER=${19}
TEST=${20}
AUX_TYPE=${21}
PREFIX=${22}

if [[ $PREFIX == 1 ]]; then
  PREFIX_AUG='--add_task_prefix'
elif [[ $PREFIX == 2 ]]; then
  PREFIX_AUG='--add_task_prefix --add_lang_ids'
fi

if [[ $PREFIX != 0 ]]; then
  PREFIX_NAME='_prefix'${PREFIX}
fi

if [[ $AUX_TYPE != 0 ]]; then
  AUX_NAME='_aux'${AUX_TYPE}
fi

if [[ $TEST == 1 ]]; then
  TEST_AUG='--do_test'
else
  TEST_AUG='--do_train --do_eval --do_eval_bleu --do_test'
fi

if [[ $DATA_NUM == -1 ]]; then
  DATA_TAG='all'
else
  DATA_TAG=$DATA_NUM
  EPOCH=1
fi

EFF_BS=$((${BS}*${GRADIENT_STEP}))
if [[ ${TASK} == 'multi_task' || ${TASK} == 'multi_auxiliary' || ${TASK} == 'summarize_auxiliary' ]]; then
  FULL_MODEL_TAG=${MODEL_TAG}_${DATA_TAG}_lr${LR}_s${23}_a${AUX_PER}${AUX_NAME}${PREFIX_NAME}
else
  FULL_MODEL_TAG=${MODEL_TAG}_${DATA_TAG}_lr${LR}_bs${EFF_BS}_src${SRC_LEN}_trg${TRG_LEN}_pat${PATIENCE}_e${EPOCH}
fi


if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${FULL_MODEL_TAG}
else
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
fi

CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

if [[ $MODEL_TAG == roberta ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=roberta-base
elif [[ $MODEL_TAG == codebert ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=microsoft/codebert-base
elif [[ $MODEL_TAG == bart_base ]]; then
  MODEL_TYPE=bart
  TOKENIZER=facebook/bart-base
  MODEL_PATH=facebook/bart-base
elif [[ $MODEL_TAG == codet5_small ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-small
  MODEL_PATH=Salesforce/codet5-small
elif [[ $MODEL_TAG == codet5_base ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-base
  MODEL_PATH=Salesforce/codet5-base
elif [[ $MODEL_TAG == cotext ]]; then
  MODEL_TYPE=t5
  TOKENIZER=razent/cotext-2-cc
  MODEL_PATH=razent/cotext-2-cc
elif [[ $MODEL_TAG == t5_base ]]; then
  MODEL_TYPE=t5
  TOKENIZER=t5-base
  MODEL_PATH=t5-base
fi


if [[ ${TASK} == 'multi_task' ]]; then
  RUN_FN=${WORKDIR}/run_multi_gen_cont.py
  MULTI_TASK_AUG='--max_steps '${23}' --save_steps '${24}' --log_steps '${25}
elif [[ ${TASK} == 'clone' ]]; then
  RUN_FN=${WORKDIR}/run_clone_cont.py
elif [[ ${TASK} == 'defect' ]] && [[ ${MODEL_TYPE} == 'roberta' ||  ${MODEL_TYPE} == 'bart' ]]; then
  RUN_FN=${WORKDIR}/run_defect_cont.py
elif [[ ${TASK} == 'multi_auxiliary' ]]; then
  RUN_FN=${WORKDIR}/run_multi_gen_aux.py
  MULTI_TASK_AUG='--max_steps '${23}' --save_steps '${24}' --log_steps '${25}' --aux_type '${AUX_TYPE}
  elif [[ ${TASK} == 'summarize_auxiliary' ]]; then
  RUN_FN=${WORKDIR}/run_summarize_aux.py
  MULTI_TASK_AUG='--max_steps '${23}' --save_steps '${24}' --log_steps '${25}' --aux_type '${AUX_TYPE}
else
  RUN_FN=${WORKDIR}/run_gen_cont.py
fi

if [[ ${LOAD_PATH} != 'no' ]]; then
  LOAD_ARG='--cont_model_path '${LOAD_PATH}
fi


cmd="CUDA_VISIBLE_DEVICES=${GPU} \
  python ${RUN_FN}  \
  ${TEST_AUG} ${MULTI_TASK_AUG} ${PREFIX_AUG} --gradient_accumulation_steps ${GRADIENT_STEP} \
  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM} --aux_percentage ${AUX_PER} \
  --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
  --tokenizer_name=${TOKENIZER}  --model_name_or_path=${MODEL_PATH} --data_dir ${DATADIR}  \
  --cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --save_last_checkpoints --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} ${LOAD_ARG} \
  --train_batch_size ${BS} --eval_batch_size ${EVAL_BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
  2>&1 | tee ${LOG}"

echo ${cmd}
eval ${cmd}
