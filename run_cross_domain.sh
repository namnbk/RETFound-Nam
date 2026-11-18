#!/bin/bash
# In-domain + cross-domain DME-vs-NORMAL experiment (OCT2017 <-> OCTDL).
#
#   train oct2017(train,val) -> eval oct2017(test)            [in-domain]
#   train oct2017(train,val) -> eval octdl(train+val+test)    [cross, whole OCTDL]
#   train octdl(train,val)   -> eval octdl(test)              [in-domain]
#   train octdl(train,val)   -> eval oct2017(test)            [cross]
# each for both adaptation = lp (linear probe) and finetune. One run (seed 0) each.
#
# Usage:  bash run_cross_domain.sh            # train (skips finished) + evaluate + summarize
#         SKIP_TRAIN=1 bash run_cross_domain.sh   # evaluation + summary only
#         ADAPTATIONS="lp" bash run_cross_domain.sh   # subset
set -e

MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
FINETUNE="RETFound_dinov2_meh"
OUT="output_dir"
EVAL_OUT="${OUT}/cross_domain_2cls"
ADAPTATIONS="${ADAPTATIONS:-lp finetune}"   # lp first: it is much faster

OCT2017_PATH="./OCT2017"
OCTDL_PATH="./octdl"

UNAME_S="$(uname -s 2>/dev/null || echo Unknown)"
if [ "$OS" = "Windows_NT" ] || echo "$UNAME_S" | grep -Eq "(MINGW|MSYS|CYGWIN)"; then
  LAUNCH="python -u"
else
  LAUNCH="torchrun --nproc_per_node=1 --master_port=48766"
fi

train() {  # $1=dataset tag, $2=data path, $3=adaptation
  local task="${MODEL_ARCH}_${1}_2cls_${3}"
  if [ -f "${OUT}/${task}/checkpoint-best.pth" ]; then
    echo "[train] ${task}: checkpoint exists, skipping"; return
  fi
  echo "[train] ${task}"
  $LAUNCH main_finetune.py \
    --model "${MODEL}" --model_arch "${MODEL_ARCH}" --finetune "${FINETUNE}" \
    --savemodel --global_pool --batch_size 24 --world_size 1 --epochs 50 \
    --nb_classes 2 --data_path "$2" --input_size 224 \
    --task "${task}" --adaptation "$3" --output_dir "${OUT}"
}

evaluate() {  # $1=train tag, $2=adaptation, $3=test data path, $4=test name, $5...=splits
  local task="${MODEL_ARCH}_${1}_2cls_${2}"
  local path="$3" name="$4"; shift 4
  python cross_domain_eval.py \
    --checkpoint "${OUT}/${task}/checkpoint-best.pth" \
    --data_path "${path}" --splits "$@" --test_name "${name}" --out_root "${EVAL_OUT}"
}

for ADAPT in ${ADAPTATIONS}; do
  if [ -z "${SKIP_TRAIN}" ]; then
    train oct2017 "${OCT2017_PATH}" "${ADAPT}"
    train octdl   "${OCTDL_PATH}"   "${ADAPT}"
  fi
  evaluate oct2017 "${ADAPT}" "${OCT2017_PATH}" oct2017_test      test
  evaluate oct2017 "${ADAPT}" "${OCTDL_PATH}"   octdl_all         train val test
  evaluate octdl   "${ADAPT}" "${OCTDL_PATH}"   octdl_test        test
  evaluate octdl   "${ADAPT}" "${OCT2017_PATH}" oct2017_test      test
done

python summarize_cross_domain.py "${EVAL_OUT}"
