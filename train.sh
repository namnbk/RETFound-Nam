# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
FINETUNE="RETFound_dinov2_meh"

# ==== Data settings ====
# change the dataset name and corresponding class number
DATASET="OCT2017"
NUM_CLASS=4
data_path="C:\Users\OCT3\Documents\Nam\AI Diagnosis\Kermany et al_\OCT2017\OCT2017"
task="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

# On Windows, torchrun may fail (no libuv). Fallback to single-process python.
UNAME_S="$(uname -s 2>/dev/null || echo Unknown)"
if [ "$OS" = "Windows_NT" ] || echo "$UNAME_S" | grep -Eq "(MINGW|MSYS|CYGWIN)"; then
  LAUNCH="python -u"
else
  LAUNCH="torchrun --nproc_per_node=1 --master_port=48766"
fi

$LAUNCH main_finetune.py \
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