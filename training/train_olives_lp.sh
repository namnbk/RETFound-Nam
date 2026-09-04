# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="lp"
MODEL="RETFound_mae"
MODEL_ARCH="retfound_mae"
FINETUNE="RETFound_mae_natureOCT"

# ==== Data settings ====
# change the dataset name and corresponding class number
DATASET="OLIVES"
NUM_CLASS=2
data_path="./olives_imagefolder"
task="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

# On Windows, torchrun may fail (no libuv). Fallback to single-process python.
UNAME_S="$(uname -s 2>/dev/null || echo Unknown)"
if [ "$OS" = "Windows_NT" ] || echo "$UNAME_S" | grep -Eq "(MINGW|MSYS|CYGWIN)"; then
  LAUNCH="python -u"
else
  LAUNCH="torchrun --nproc_per_node=1 --master_port=48766"
fi

$LAUNCH training/main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --finetune "${FINETUNE}" \
  --savemodel \
  --global_pool \
  --batch_size 24 \
  --world_size 1 \
  --epochs 50 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${data_path}" \
  --input_size 224 \
  --task "${task}" \
  --adaptation "${ADAPTATION}"
