# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="RETFound_mae"
MODEL_ARCH="retfound_mae"
FINETUNE="RETFound_mae_natureOCT"

# ==== Data settings ====
DATASET="OLIVESBiomarkers"
NUM_CLASS=6
LABEL_COLUMNS="IRHRF,PAVF,FAVF,IRF,DRTME,VD"
data_path="./olives_biomarker_imagefolder"
labels_csv="./data/olives_split/olives_biomarker_manifest.csv"
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
  --batch_size 12 \
  --accum_iter 2 \
  --world_size 1 \
  --epochs 50 \
  --nb_classes "${NUM_CLASS}" \
  --multilabel \
  --data_path "${data_path}" \
  --labels_csv "${labels_csv}" \
  --label_columns "${LABEL_COLUMNS}" \
  --input_size 224 \
  --task "${task}" \
  --adaptation "${ADAPTATION}"
