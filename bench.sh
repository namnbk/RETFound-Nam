# filepath: bench.sh
# ==== Model settings (same as train.sh) ====
ADAPTATION="finetune"
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"

# ==== Data settings (same as train.sh) ====
DATASET="octdl"
NUM_CLASS=7
data_path="./${DATASET}"
task="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

# Path to the trained checkpoint
CKPT="./output_dir/${task}/checkpoint-best.pth"

# On Windows, use single-process python
UNAME_S="$(uname -s 2>/dev/null || echo Unknown)"
if [ "$OS" = "Windows_NT" ] || echo "$UNAME_S" | grep -Eq "(MINGW|MSYS|CYGWIN)"; then
  LAUNCH="python -u"
else
  LAUNCH="python -u"
fi

# Run single-image latency benchmark on the test set
$LAUNCH bench_infer.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --global_pool \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${data_path}" \
  --input_size 224 \
  --task "${task}" \
  --resume "${CKPT}" \
  --bench_warmup 20 --bench_iters 100
