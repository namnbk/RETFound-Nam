# Method: RETFound Linear Probing on OLIVES + Grad-CAM Explanation

**Fine-tuning.** The classifier is `RETFound_mae` (ViT-L/16, 24 transformer
blocks, embedding dim $D=1024$), pretrained with masked autoencoding on a
large natural + OCT image corpus (`RETFound_mae_natureOCT`). It is adapted to
2-class DME-vs-DR classification on the OLIVES dataset via **linear probing**
(`train_olives_lp.sh` → `main_finetune.py --adaptation lp`): every backbone
parameter is frozen (`requires_grad = ("head" in name)`), so only the linear
classification head is trained on top of the frozen, mean-pooled,
layer-normalized patch tokens $A \in \mathbb{R}^{196\times 1024}$ from the last
block —

$$z = W_{\text{head}}\cdot\text{LN}\!\Big(\text{LN}\big(\tfrac{1}{N}\textstyle\sum_n A_n\big)\Big) + b_{\text{head}}$$

Data is split **by patient ID**, not by B-scan, using
`olives_split/splits.json` (train/val/test disjoint patient sets), so that
scans from the same eye/patient never leak across splits: 60 patients →
train, 9 → val, 18 → test (87 total). Images are materialized into an
ImageFolder layout (`olives_imagefolder/`) with one subfolder per class,
giving, after all B-scans for each patient are pooled into their split:

| Split | DME | DR | Total |
|---|---|---|---|
| train | 31,246 | 24,791 | 56,037 |
| val | 2,842 | 4,067 | 6,909 |
| test | 7,840 | 8,036 | 15,876 |

The best checkpoint by validation accuracy is saved to
`output_dir/retfound_mae_OLIVES_lp/checkpoint-best.pth` and is the only
trained classifier this repo currently ships.

**Explanation via Grad-CAM.** Since a ViT has no spatial conv feature map,
Grad-CAM (Selvaraju et al., 2017) is adapted token-wise: the last block's
patch tokens $A$ play the role the last conv layer's feature map plays in a
CNN, with the token index replacing the pixel index. Backpropagating the
target class logit $z_y$ back to $A$ gives gradients that are averaged over
tokens into a per-channel importance weight, then used to collapse channels
into a per-token relevance score, clipped at zero:

$$\alpha_d = \tfrac{1}{N}\sum_n \frac{\partial z_y}{\partial A_{n,d}} \qquad L_n = \text{ReLU}\Big(\sum_d \alpha_d A_{n,d}\Big)$$

$L \in \mathbb{R}^{196}$ is min-max normalized, reshaped to the $14\times14$
patch grid, bicubic-upsampled to $224\times224$, and overlaid on the input
scan. Because of the ReLU, a dim/blue region means "not positive evidence for
this class," not "evidence against it" — see `GRADCAM_METHODS.md` for the
full derivation, caveats, and code-to-math mapping (`grad_cam_retfound.py`).

**Exposure as MCP tools.** `mcp_servers/retinal_ai/server.py` wraps
`grad_cam_retfound.py`'s `load_classifier`/`run_gradcam` so an LLM coding
agent can classify a single B-scan, screen an entire OCT volume for the most
diagnostically relevant slices in one call, or generate a Grad-CAM overlay for
a specific slice — returning the heatmap as an inline image the agent can see
directly, rather than a file path it must separately open.
`mcp_servers/retinal_data/server.py` provides the matching data-access tools
(e.g. reading slices out of `olives_zarr/`) that these calls operate on. Both
tools report only class-logit attribution: they never return ground-truth
OLIVES labels and never assert that a heatmap corresponds to a specific
clinical biomarker.
