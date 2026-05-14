# MUSER: Memory-Augmented Multimodal Sequential Recommendation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

This repository is the official implementation of the paper: **"MUSER: Memory-Augmented Multimodal Sequential Recommendation"**.

MUSER is a robust framework designed to leverage Multimodal Large Language Models (MLLMs) for 
recommendation tasks. It integrates visual perception with textual reasoning to provide accurate 
and explainable recommendations. We support a wide variety of state-of-the-art backbones, 
including **Qwen2.5-VL**, **LLaVA v1.6**, and **LFM2-VL-450M**.

## 📂 Project Structure

```text
MUSER/
├── data/                       # Dataset management
│   ├── microlens/              # MicroLens dataset & preprocessing
│   ├── movielens/              # MovieLens dataset
│   └── ninerec/                # NineRec dataset
├── train/                      # Training implementations for different backbones
│   ├── DeepSeek-OCR/           # DeepSeek-OCR integration
│   ├── InternVL2-1B/           # InternVL2 (1B) implementation
│   ├── LFM2-VL-450M/           # LFM2-VL implementation
│   ├── llava-interleave-*/     # LLaVA Interleave variants
│   ├── llava-v1.6-*/           # LLaVA v1.6 variants (Mistral, Vicuna)
│   └── Qwen2.5-VL-3B-Instruct/ # Qwen2.5-VL implementation
├── Inference/                  # Inference scripts (Generation)
│   ├── microlens/
│   ├── movielens/
│   └── ninerec/
├── test/                       # Evaluation scripts (Metrics)
│   ├── microlens/
│   ├── movielens/
│   └── ninerec/
├── save/                       # Checkpoints and logs
├── environment.yml             # Conda environment config
├── requirements.txt            # Pip requirements
└── appendix.pdf                # Supplementary materials
```

## 🛠️ Environment Setup
Please ensure you have Python 3.9+ and CUDA installed. You can set up the environment using Conda:
```
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate muser

# 3. Install additional dependencies (if necessary)
pip install -r requirements.txt
```

## 📊 Data Preparation
We utilize three multimodal datasets: MicroLens, MovieLens, and NineRec.

Preprocessing
Navigate to the preprocessing directory of the specific dataset to process raw data into training pairs. For example, for MicroLens:

```text
cd data/microlens/preprocessing

# Option A: Run the Jupyter Notebook
jupyter notebook preprocessing.ipynb

# Option B: Ensure the following files are generated in the directory
# - train_pairs.csv
# - val_pairs.csv
# - test_pairs.csv
# - user_items_negs.tsv
```

🚀 Training
The training code is modularized by the backbone model architecture. Navigate to the train/ directory and select the model you wish to fine-tune.

Supported Backbones
Qwen Series: Qwen2.5-VL-3B-Instruct

LLaVA Series: v1.6-Mistral-7b, v1.6-Vicuna-13b, v1.6-34b, Interleave-Qwen

InternVL Series: InternVL2-1B

Others: DeepSeek-OCR, LFM2-VL

Example Training Command
To train the Qwen2.5-VL model on the MicroLens dataset:

```
cd train/Qwen2.5-VL-3B-Instruct

# Run the training script (Modify arguments based on your hardware)
python train.py \
    --dataset microlens \
    --data_path ../../data/microlens \
    --output_dir ../../save/Qwen2.5-microlens \
    --batch_size 16 \
    --epochs 5
   ```
(Note: Adjust the script name train.py if the actual file is named differently, e.g., finetune.py or main.py)

## 🔮 Inference & Evaluation
Evaluation is a two-step process: Inference (generating explanations/recommendations) and Testing (calculating metrics).

1. Inference
Generate responses using the trained checkpoints:

```
cd Inference/microlens
python inference.py --model_path ../../save/Qwen2.5-microlens --output_file prediction.json

```

2. Testing
Evaluate the generated responses against ground truth:
```
cd test/microlens
python test.py --prediction_file ../../Inference/microlens/prediction.json
```
