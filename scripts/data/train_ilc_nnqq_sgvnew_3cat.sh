#!/bin/bash

set -x

#source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
#DATADIR=${DATADIR_JetClass}
#[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass'

DATADIR='/data/suehara/mldata/flavortag/20250711_ilc_nnqq_sgv_10m'
# DATADIR='/data/suehara/mldata/flavortag/20250614_ilc_nnqq_sgv_10m'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=20
samples_per_epoch=$((2400000 / $NGPUS))
samples_per_epoch_val=$((150000))
dataopts="--num-workers 1 --fetch-step 0.01"

gpu_index=$1
#gpu_index=3
# PN, PFN, PCNN, ParT
# model=$1
[[ -z ${model} ]] && model="ParT"

if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformerTagger_renew.py --use-amp"
    batchopts="--batch-size 256 --start-lr 1e-3"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "full"
# FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# currently only Pythia
yaml_name="ilc_nnqq_sgv_3category_test"

$CMD \
    --data-train \
    "nnbb:${DATADIR}/*_bb_train800k.root" \
    "nncc:${DATADIR}/*_cc_train800k.root" \
    "nndd:${DATADIR}/*_dd_train800k.root" \
    --data-val \
    "nnbb:${DATADIR}/*_bb_val50k.root" \
    "nncc:${DATADIR}/*_cc_val50k.root" \
    "nndd:${DATADIR}/*_dd_val50k.root" \
    --data-test \
    "nnbb:${DATADIR}/*_bb_test150k.root" \
    "nncc:${DATADIR}/*_cc_test150k.root" \
    "nndd:${DATADIR}/*_dd_test150k.root" \
    --data-config data/${yaml_name}.yaml --network-config $modelopts \
    --model-prefix training/unprocessed/${yaml_name}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus $gpu_index \
    --optimizer ranger --log logs/unprocessed/${yaml_name}/${yaml_name}.log --predict-output /data/suehara/part/takahiro/training/unprocessed/ilc_nnqq_sgv_3category_test/predict_output_1m1/pred.root \
    --tensorboard ILC_${yaml_name}_${suffix} \
    --gpus "1,2,3" \
    "${@:3}"
