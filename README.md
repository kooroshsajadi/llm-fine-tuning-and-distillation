# LLM Fine-Tuning and Distillation 🚀

**LLM Fine-Tuning and Distillation** is an under-development project aimed at training and deploying lightweight, efficient large language models (LLMs) using knowledge distillation and advanced model compression techniques. It builds on the Hugging Face Transformers ecosystem for seamless integration, flexibility, and modularity. The project aims to support fine-tuning of various LLM architectures (causal LM, masked LM, and seq2seq) with efficient adapters like LoRA and QLoRA, enabling resource-efficient training on diverse hardware setups.

Key highlights include dynamic adaptation to hardware (e.g., automatic QLoRA for NVIDIA GPUs or fallback to LoRA on CPUs), optimized data types like bfloat16 for faster CPU performance, comprehensive evaluation metrics tailored for seq2seq tasks, and modular components for training, inference, and evaluation.

## 📌 Project Goals

- Distill knowledge from large teacher models into smaller, efficient student models using techniques like logit matching.
- Implement adapter-based fine-tuning (LoRA/QLoRA) for parameter-efficient updates without full model retraining.
- Support fast inference with reduced memory and computational overhead, ideal for deployment on edge devices or limited resources.
- Provide reproducible, modular code for easy experimentation, including hardware-aware optimizations and robust evaluation.

## ⚙️ Features

### 🔧 Flexible Model Support:

Currently compatible with seq2seq models (e.g., T5 or MarianMT), aiming for compatibility with causal LM (e.g., GPT-2) and masked LM (e.g., BERT). Automatically loads and configures models via [ModelLoader](./src/core/model_loader.py), which handles device mapping ("auto" for CPU/GPU detection), quantization, and dtype selection (e.g., float16 for GPUs, bfloat16 for Intel CPUs like ARC GPUs).

### 🧠 Advanced Fine-Tuning with QLoRA/LoRA:

- **Dynamic hardware adaptation**: Automatically detects NVIDIA GPUs for QLoRA (4-bit quantization with BitsAndBytes for memory efficiency, reducing footprint by up to 4x while maintaining performance). Falls back to standard LoRA on CPUs or non-compatible hardware.
- **Optimized for efficiency**: Uses bfloat16 on compatible CPUs (e.g., Intel ARC via IPEX optimization) for faster training and inference without precision loss. Gradient checkpointing and accumulation steps further reduce memory usage during training.
- **Configurable LoRA parameters**: Rank (default 8), alpha (default 32), dropout (default 0.05), and target modules auto-detected based on model type (e.g., attention layers for seq2seq).
- Supports training with custom datasets, including preprocessing via Hugging Face Datasets and tokenizers.

### 🗃️ Dataset Handling:

Modular preprocessing with [prepare_tokenized_dataset](./src/data/data_preparation.py) and [prepare_dataset_dict](./src/data/data_preparation.py) for creating DatasetDicts (train/validation/test splits). Handles tokenization, padding, and truncation with max sequence lengths.

### 🎯 Knowledge Distillation and Compression:

Focuses on logit-based supervision for distilling teacher knowledge into students. Planned expansions include additional compression methods like pruning and quantization-aware training.

### 📊 Comprehensive Evaluation Metrics:

- **Tailored for seq2seq models**: Computes `ROUGE` (1/2/L with stemming), `BLEU` (corpus-level with smoothing), and `BERTScore` (F1 with multilingual support, rescaled with baseline).
- Integrated into trainers via [HFMetricHelper](./src/utils/metrics_utils.py) for automatic metric calculation during evaluation and inference.

### 📦 Safe Model Management:

Uses `safetensors` for secure saving/loading with integrity checks. Models and tokenizers are saved separately for easy reuse.

### 🧪 Modular Design

- Separate modules for fine-tuning ([QLoRAFineTuner](./src/fine_tuning/qlora_fine_tuner.py)), model loading ([ModelLoader](./src/core/model_loader.py)), metrics ([HFMetricHelper](./src/utils/metrics_utils.py)), and [data preparation](./src/data/data_preparation.py).
- [Inference script](./src/core/inference.py) for peforming evaluation on the test dataset and computing metrics.
- Logging utilities for detailed tracking, including memory usage (via psutil) and gradient verification.

### 🔍 Hardware Optimizations

- Automatic device detection and dtype selection for peak performance (e.g., fp16 on CUDA, bf16 on XPU).
- **QLoRA-specific features**: Paged AdamW 8-bit optimizer for low-memory training on GPUs.
- **Model profiling**: Logs parameter counts (total/trainable), memory estimates, and quantization status for transparency.

## 🚧 Status

> **Note**: This project is still **under active development**. Expect changes in structure, functionality, and APIs in the near future. Currently focused on seq2seq fine-tuning; full distillation pipeline coming soon.

## 📄 License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (the "License"). You may not use, copy, modify, or distribute this project except in compliance with the License. A copy of the License is included in the [LICENSE](./LICENSE) file in this repository.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations.

**Note**: This license applies to the source code in this repository. Trained models and datasets may be subject to separate licensing terms due to their dependencies or data sources. Please refer to the respective documentation for details.
