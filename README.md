# Event-Biased ViT for Object Detection

A modified **ViT-B/16** that injects event camera data (from an event sensor) into the last 4 transformer blocks as an additive attention bias — without changing any pretrained weights.

## Architecture

```
Pretrained ViT-B/16 (frozen)
  └─ Last 4 blocks: Attention logits += alpha * event_density_j
        alpha: learned scalar per block, init 0.0
        event_density_j: log-normalized event count per patch (B, num_patches)
Detection Head (trainable)
  └─ Linear(768, 256) → ReLU → Linear(256, 5)  per patch
        output: (objectness, cx, cy, w, h)
Trainable params: 4 alpha scalars + DetectionHead only
```

## Dataset: PEOD

Pixel-aligned Event-RGB dataset for Object Detection.  
Provides synchronized RGB frames + raw events (x, y, t, polarity) in HDF5 format  
with bounding box annotations and lighting condition metadata.

## Local Setup (Mac / Linux)

```bash
# 1. Clone
git clone <your-repo-url>
cd project

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place PEOD data
mkdir -p data/peod/train data/peod/val
# copy *.h5 sequence files into data/peod/train/ and data/peod/val/
```

## Running

```bash
# Train
python train.py --data_root data/peod --epochs 20 --batch_size 8

# Evaluate (mAP per lighting condition)
python evaluate.py --data_root data/peod --checkpoint checkpoints/ckpt_epoch020.pt

# Visualize attention maps
python visualize.py \
  --image data/sample.png \
  --events_npy data/sample_events.npy \
  --out attention_comparison.png
```

**VS Code:** open the project folder, select your `.venv` Python interpreter,  
then use **Run → Start Debugging** (F5) — launch configs for all three scripts are in `.vscode/launch.json`.

## Lab Server Setup

```bash
# 1. Clone on server
git clone <your-repo-url>
cd project

# 2. Virtual env (or use conda)
python3 -m venv .venv && source .venv/bin/activate

# 3. Install (CUDA build of PyTorch if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 4. Run with more workers / larger batch
python train.py \
  --data_root /path/to/peod \
  --epochs 50 \
  --batch_size 32 \
  --num_workers 8
```

For SLURM clusters, wrap the train command in an `sbatch` script.

## File Structure

```
project/
├── event_utils.py   # raw events → per-patch density (B, num_patches)
├── dataset.py       # PEODDataset + collate_fn
├── model.py         # EventBiasAttention, DetectionHead, build_model, set_event_bias
├── train.py         # training loop, target assignment, detection loss
├── evaluate.py      # mAP@0.5 split by lighting condition
├── visualize.py     # 4-panel attention comparison figure
└── requirements.txt
```

## HDF5 Layout Expected by PEODDataset

```
sequence.h5
├── images/frame_0          (H, W, 3) uint8
├── events/frame_0/x        float32 array
├── events/frame_0/y        float32 array
├── events/frame_0/p        int8 array
├── labels/frame_0/boxes    (N, 4) float32, normalized [x1,y1,x2,y2]
├── labels/frame_0/labels   (N,)  int32
└── metadata/condition      per-frame string: normal | low_light | motion_blur
```
