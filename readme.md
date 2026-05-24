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
````

## 🛠️ Environment Setup

Please ensure you have Python 3.9+ and CUDA installed. You can set up the environment using Conda:

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate muser

# 3. Install additional dependencies (if necessary)
pip install -r requirements.txt
```

## 📊 Data Preparation

We utilize three multimodal datasets: MicroLens, MovieLens, and NineRec.

### Preprocessing

Navigate to the preprocessing directory of the specific dataset to process raw data into training pairs. For example, for MicroLens:

```bash
cd data/microlens/preprocessing

# Option A: Run the Jupyter Notebook
jupyter notebook preprocessing.ipynb

# Option B: Ensure the following files are generated in the directory
# - train_pairs.csv
# - val_pairs.csv
# - test_pairs.csv
# - user_items_negs.tsv
```

## 🚀 Training

The training code is modularized by the backbone model architecture. Navigate to the `train/` directory and select the model you wish to fine-tune.

### Supported Backbones

* Qwen Series: Qwen2.5-VL-3B-Instruct
* LLaVA Series: v1.6-Mistral-7B, v1.6-Vicuna-13B, v1.6-34B, Interleave-Qwen
* InternVL Series: InternVL2-1B
* Others: DeepSeek-OCR, LFM2-VL

### Example Training Command

To train the Qwen2.5-VL model on the MicroLens dataset:

```bash
cd train/Qwen2.5-VL-3B-Instruct

# Run the training script. Modify arguments based on your hardware.
python train.py \
    --dataset microlens \
    --data_path ../../data/microlens \
    --output_dir ../../save/Qwen2.5-microlens \
    --batch_size 16 \
    --epochs 5
```

Note: Adjust the script name `train.py` if the actual file is named differently, e.g., `finetune.py` or `main.py`.

## 🔮 Inference & Evaluation

Evaluation is a two-step process: inference, which generates recommendations and explanations, and testing, which calculates ranking and explanation metrics.

### 1. Inference

Generate responses using the trained checkpoints:

```bash
cd Inference/microlens
python inference.py --model_path ../../save/Qwen2.5-microlens --output_file prediction.json
```

### 2. Testing

Evaluate the generated responses against ground truth:

```bash
cd test/microlens
python test.py --prediction_file ../../Inference/microlens/prediction.json
```

## ⚙️ Baseline Implementation Details

The following table summarizes the implementation details and hyperparameter configurations for all baselines. To ensure reproducible and fair benchmarking, we standardize common hyperparameters where applicable, while adhering to the optimal model-specific configurations reported in the original literature.

Notation: `d` denotes embedding dimension, `l` denotes learning rate, `b` denotes batch size, `L` denotes the number of layers or blocks, and `H` denotes the number of attention heads.

| Model        | Key Hyperparameter Settings & Implementation Details                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| GRU4Rec      | `d=64`, `l=1e-3`, `b=1024`, hidden units=128. Loss: BPR/Cross-Entropy. Optimizer: Adam.                                         |
| SASRec       | `d=64`, `l=1e-3`, `b=1024`, `L=2`, `H=1`, dropout=0.5, max_len=200. Positional Embedding: Learnable.                            |
| GRU4Rec_F    | Extends GRU4Rec. Visual features projected to `d` via MLP. Concatenation strategy: Early fusion.                                |
| SASRec_F     | Extends SASRec. Visual features projected to `d`. Fusion: Added to item embeddings before self-attention.                       |
| SIMTIER-MAKE | `d=64`, `l=1e-3`, `b=1024`. Hierarchical retrieval. Index size=1000, retrieve top-K=50.                                         |
| MUSE         | `d=64`, `l=5e-4`, `b=1024`, `L=2`. Multi-granularity intent extraction. Loss: BPR + Contrastive.                                |
| MMGCN        | `d=64`, `l=1e-3`, `b=1024`, GCN layers `L=2`. Aggregation: Mean/Max Pooling. Modalities: Visual + ID.                           |
| MGAT         | `d=64`, `l=5e-4`, `b=1024`, GAT layers `L=2`, heads `H=4`. Attention dropout=0.1.                                               |
| LGMRec       | `d=64`, `l=1e-3`, `b=1024`, LightGCN layers `L=3`. Modality dropout=0.1. Optimizer: Adam.                                       |
| EVEN         | `d=64`, `l=1e-3`. GCN layers `L=2`. Alignment regularization `lambda_align=1e-2`, temperature `tau=0.1`.                        |
| LLaVA        | Zero-shot setting. Temperature=0.1, max_new_tokens=512. Prompt: "Recommend a video based on..." No training.                    |
| TALLREC      | Backbone: LLaMA-7B. LoRA: `r=8`, `alpha=16`. `l=1e-4`, `b=128` with gradient accumulation. Epochs=3. Instruction tuning format. |
| MLLM-MSR     | Backbone: LLaVA-1.6-7B. LoRA: `r=8`, `alpha=16`, targets=[q,k,v,o]. `l=2e-4`. Recurrent Summary Window=3.                       |
| NRT          | `d=300` for word embeddings, `d_id=64`. GRU hidden=400. `l=1e-3`. Regularization weight `lambda=1e-4`.                          |
| Attn2Seq     | `d=512`. Encoder: MLP. Decoder: LSTM with Attention. `l=1e-3`. User/item visual projection included.                            |
| PETER        | `d=512`, `L=2`, `H=2`. `l=1e-4`. Masking: Peter-Mask. Loss: NLL for generation + CE for recommendation.                         |
| XRec         | Backbone: LLaMA-7B with soft prompting or adapter. Prompt length=10. `l=1e-3` for adapters.                                     |
## 🧠 Algorithmic Details

This section presents the algorithmic workflows of the two core agents in MUSER: the User Agent (UA) and the Recommendation Agent (RA). The UA evolves long-term and short-term user memory, while the RA consumes the evolved memory state for recommendation training and inference.

### User Agent Memory Evolution
![User Agent (UA) Memory Evolution in MUSER](./algorithm1.png)

### Recommendation Agent Training and Inference
![Recommendation Agent (RA) Training and Inference in MUSER](./algorithm2.png)

### GPTScore Evaluation

For RQ2, we use GPTScore to evaluate the semantic quality of generated explanations. To make the LLM-as-judge protocol transparent, we use GPT-4o-2024-08-06 through the OpenAI API with temperature set to 0. For each test instance, the model-generated rationale and the ground-truth rationale are provided to a fixed judge prompt.

The judge scores each explanation from five aspects: semantic alignment, factual faithfulness, preference grounding, specificity and insight, and coherence/fluency. Each aspect is assigned a score from 1 to 5. We first average the five aspect scores for each instance, normalize the result to \([0,1]\) by \((s_i - 1) / 4\), and then average the normalized scores over all test samples to obtain the final dataset-level GPTScore.

Therefore, a reported value such as 0.8427 is the average normalized GPTScore over the full test set, rather than a single direct score produced by GPT-4.

Please see `gptscore_judge.py` for the detailed implementation and the full judge prompt.

## 📌 Notes

* The appendix provides additional implementation details, algorithmic descriptions, and qualitative analyses.
* Please adjust dataset paths, backbone paths, and checkpoint paths according to your local environment.
* For large MLLM backbones, gradient accumulation and distributed training are recommended.

```
```
