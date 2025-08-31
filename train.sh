# ==== Model settings ====
ADAPTATION="finetune"
MODEL="Dinov2"
MODEL_ARCH="dinov2_vitl14"
FINETUNE="dinov2_vitl14_pretrain.pth"

# ==== Data settings ====
DATASET="MESSIDOR2"
NUM_CLASS=5

data_path="/home/jupyter/public_dataset/${DATASET}"
task="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
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
