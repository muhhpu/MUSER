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

| Category     | Model        | Key Hyperparameter Settings & Implementation Details                                                                            |
| ------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Seq & Search | GRU4Rec      | `d=64`, `l=1e-3`, `b=1024`, hidden units=128. Loss: BPR/Cross-Entropy. Optimizer: Adam.                                         |
| Seq & Search | SASRec       | `d=64`, `l=1e-3`, `b=1024`, `L=2`, `H=1`, dropout=0.5, max_len=200. Positional Embedding: Learnable.                            |
| Seq & Search | GRU4Rec_F    | Extends GRU4Rec. Visual features projected to `d` via MLP. Concatenation strategy: Early fusion.                                |
| Seq & Search | SASRec_F     | Extends SASRec. Visual features projected to `d`. Fusion: Added to item embeddings before self-attention.                       |
| Seq & Search | SIMTIER-MAKE | `d=64`, `l=1e-3`, `b=1024`. Hierarchical retrieval. Index size=1000, retrieve top-K=50.                                         |
| Seq & Search | MUSE         | `d=64`, `l=5e-4`, `b=1024`, `L=2`. Multi-granularity intent extraction. Loss: BPR + Contrastive.                                |
| Graph-based  | MMGCN        | `d=64`, `l=1e-3`, `b=1024`, GCN layers `L=2`. Aggregation: Mean/Max Pooling. Modalities: Visual + ID.                           |
| Graph-based  | MGAT         | `d=64`, `l=5e-4`, `b=1024`, GAT layers `L=2`, heads `H=4`. Attention dropout=0.1.                                               |
| Graph-based  | LGMRec       | `d=64`, `l=1e-3`, `b=1024`, LightGCN layers `L=3`. Modality dropout=0.1. Optimizer: Adam.                                       |
| Graph-based  | EVEN         | `d=64`, `l=1e-3`. GCN layers `L=2`. Alignment regularization `lambda_align=1e-2`, temperature `tau=0.1`.                        |
| MLLM-based   | LLaVA        | Zero-shot setting. Temperature=0.1, max_new_tokens=512. Prompt: "Recommend a video based on..." No training.                    |
| MLLM-based   | TALLREC      | Backbone: LLaMA-7B. LoRA: `r=8`, `alpha=16`. `l=1e-4`, `b=128` with gradient accumulation. Epochs=3. Instruction tuning format. |
| MLLM-based   | MLLM-MSR     | Backbone: LLaVA-1.6-7B. LoRA: `r=8`, `alpha=16`, targets=[q,k,v,o]. `l=2e-4`. Recurrent Summary Window=3.                       |
| Explainable  | NRT          | `d=300` for word embeddings, `d_id=64`. GRU hidden=400. `l=1e-3`. Regularization weight `lambda=1e-4`.                          |
| Explainable  | Attn2Seq     | `d=512`. Encoder: MLP. Decoder: LSTM with Attention. `l=1e-3`. User/item visual projection included.                            |
| Explainable  | PETER        | `d=512`, `L=2`, `H=2`. `l=1e-4`. Masking: Peter-Mask. Loss: NLL for generation + CE for recommendation.                         |
| Explainable  | XRec         | Backbone: LLaMA-7B with soft prompting or adapter. Prompt length=10. `l=1e-3` for adapters.                                     |

## 🧠 Algorithmic Details

This section presents the algorithmic workflows of the two core agents in MUSER: the User Agent (UA) and the Recommendation Agent (RA). The UA evolves long-term and short-term user memory, while the RA consumes the evolved memory state for recommendation training and inference.

### User Agent Memory Evolution

```latex
\begin{algorithm}[htbp]
\caption{User Agent (UA) Memory Evolution in MUSER}
\label{alg:muser_ua}
\SetAlgoLined
\KwIn{User multimodal interaction batches $\mathcal{B} = \{B_0, B_1, \dots, B_T\}$}
\KwOut{Evolved user memory state: $LTI_T$, $STI_T$, and personality parameter $P_\text{per}^{(T)}$}

\textbf{Initialize:} $LTI_{-1} \gets \emptyset$, $STI_{-1} \gets \emptyset$, External Storage $P_\text{per} \gets \text{null}$\;

\For{each temporal batch $t = 0, 1, \dots, T$}{
    Extract multimodal features $(\mathcal{G}_t, \mathcal{T}_t)$ from current batch $B_t$\;
    
    \eIf{$t == 0$}{
        \textit{\color{blue}// Phase: Cold-Start Encoder ($\Phi_{\text{enc}}$)}\;
        $STI_0 \gets \Phi_{\text{enc}}(\mathcal{G}_0, \mathcal{T}_0)$\;
        $LTI_0 \gets \emptyset$\;
    }{
        \eIf{$t == 1$}{
            \textit{\color{blue}// Phase: Differential Initializer ($\Phi_{\text{init}}$)}\;
            $STI_1, LTI_1, P_\text{per}^{(1)} \gets \Phi_{\text{init}}(STI_0, \mathcal{G}_1, \mathcal{T}_1)$\;
            Save $P_\text{per}^{(1)}$ to External Storage\;
        }{
            \textit{\color{blue}// Phase: Parametric Evolution ($\Phi_{\text{evolve}}$)}\;
            Retrieve $P_\text{per}^{(t-1)}$ from External Storage\;
            
            \textit{\color{purple}--- Step 1: Multimodal Perception ---}\;
            $STI_t \gets \Phi_{\text{enc}}(\mathcal{G}_t, \mathcal{T}_t)$\;
            
            \textit{\color{purple}--- Step 2: Semantic Deviation \& Deterministic Update (ICP) ---}\;
            $s_t \gets \text{LLM}_{\text{eval}}(STI_t, LTI_{t-1} \mid P_\text{per}^{(t-1)})$\;
            $\mathcal{D}_t \gets (s_t - 3) / 4$\;
            $\Delta p \gets \gamma (e^{\delta \mathcal{D}_t} - 1)$\;
            $P_\text{per}^{(t)} \gets \text{Clip}(P_\text{per}^{(t-1)} + \Delta p, 0, 1)$\;
            Save updated $P_\text{per}^{(t)}$ to External Storage\;
            
            \textit{\color{purple}--- Step 3: Structured Memory Operations ---}\;
            $LTI_t \gets \text{Op}_{\text{mem}}(LTI_{t-1}, STI_t \mid P_\text{per}^{(t)})$\;
        }
    }
}

\Return{$LTI_T, STI_T, P_\text{per}^{(T)}$}
\end{algorithm}
```

### Recommendation Agent Training and Inference

```latex
\begin{algorithm}[htbp]
\caption{Recommendation Agent (RA) Training and Inference in MUSER}
\label{alg:muser_ra}
\SetAlgoLined
\KwIn{Evolved UA memory state $\mathcal{M}_u^{(t)}=\{LTI_t, STI_t, P_\text{per}^{(t)}\}$; candidate multimodal features $(\mathcal{G}_{\text{candi}}, \mathcal{T}_{\text{candi}})$; teacher-generated target sequence $\mathcal{Y}_{tgt}=(\mathcal{A}_{gt}\oplus\mathcal{R}_{gt})$ for training}
\KwOut{Trained RA parameters $\theta$ during training; predicted decision $\hat{\mathcal{A}}$ and rationale $\hat{\mathcal{R}}$ during inference}

\textbf{Initialize:} Load student RA $\mathcal{M}_S$ with trainable LoRA parameters $\theta$\;

\textit{\color{blue}// Phase: Memory-Aware Context Construction}\;
Retrieve $\mathcal{M}_u^{(t)}=\{LTI_t, STI_t, P_\text{per}^{(t)}\}$ from UA\;
$\mathcal{X}_{\text{ctx}} \gets \{LTI_t, STI_t, P_\text{per}^{(t)}, \mathcal{G}_{\text{candi}}, \mathcal{T}_{\text{candi}}\}$\;

\eIf{\textbf{Training}}{
    \textit{\color{blue}// Phase: Semantic Trajectory Distillation}\;
    $\mathcal{Y}_{tgt} \gets \mathcal{A}_{gt} \oplus \mathcal{R}_{gt}$\;
    $P_{\theta}(\cdot \mid \mathcal{X}_{\text{ctx}}) \gets \mathcal{M}_S(\mathcal{X}_{\text{ctx}})$\;
    $\mathcal{L}_{a} \gets -\log P_{\theta}(\mathcal{A}_{gt} \mid \mathcal{X}_{\text{ctx}})$\;
    $\mathcal{L}_{r} \gets -\frac{1}{m}\sum_{k=1}^{m}\log P_{\theta}(r_k \mid r_{<k}, \mathcal{X}_{\text{ctx}}, \mathcal{A}_{gt})$\;

    \textit{\color{blue}// Phase: Dual Consistency Regularization}\;
    $S_r \gets e^{-\mathcal{L}_r}$, \quad $S_a \gets e^{-\mathcal{L}_a}$\;
    $R \gets S_r / S_a$\;
    $C_a \gets S_r(1-S_a)\cdot \mathbb{I}[R>\tau_h]$\;
    $C_r \gets (1-S_r)S_a\cdot \mathbb{I}[R<\tau_l]$\;
    $\mathcal{L}_{total} \gets (1+C_a)\mathcal{L}_a + (1+C_r)\mathcal{L}_r$\;
    $\theta \gets \theta - \eta\nabla_{\theta}\mathcal{L}_{total}$\;

    \Return{$\theta$}
}{
    \textit{\color{blue}// Phase: Autoregressive Recommendation Inference}\;
    $\hat{\mathcal{Y}} \gets \text{LLM}_{\text{dec}}(\mathcal{X}_{\text{ctx}})$\;
    $(\hat{\mathcal{A}}, \hat{\mathcal{R}}) \gets \text{Parse}(\hat{\mathcal{Y}})$\;

    \Return{$\hat{\mathcal{A}}, \hat{\mathcal{R}}$}
}
\end{algorithm}
```

## 📌 Notes

* The appendix provides additional implementation details, algorithmic descriptions, and qualitative analyses.
* Please adjust dataset paths, backbone paths, and checkpoint paths according to your local environment.
* For large MLLM backbones, gradient accumulation and distributed training are recommended.

```
```
