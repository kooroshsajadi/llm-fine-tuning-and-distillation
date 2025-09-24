# LLM Fine-Tuning and Distillation 🚀

**Distilled LLM** is an under-development project focused on training and deploying lightweight, efficient large language models through **knowledge distillation** and other model compression techniques. It leverages the Hugging Face Transformers ecosystem for flexibility and modularity.

## 📌 Project Goals

- Develop LLM fine-tuning solutions adaptable to various hardware types while exploiting their individual advancements. 
- Distill knowledge from large transformer-based language models into smaller, student models.
- Explore techniques like logit matching and adapter-based fine-tuning.
- Enable fast inference with reduced computational overhead.
- Provide clean, modular code for reproducibility and experimentation.

## ⚙️ Features

- 🔧 **Flexible architecture** for experimenting with different teacher-student models.
- 🎯 **Fine-tuning** support for custom large language models including seq2seq architectures.
- 🧠 **Knowledge distillation** with logit-based supervision.
- 🗃️ **Dataset preprocessing** with Hugging Face Datasets and tokenizers.
- 📊 **Comprehensive Evaluation** with metrics specially implemented for seq2seq models, including ROUGE, BLEU, and BERTScore to provide robust semantic and lexical quality assessment.
- 📦 **Safe saving and loading** using `safetensors` format and integrity checks.
- 🧪 **Modular design** for training, inference, and evaluation.

## 🚧 Status

> **Note:** This project is still **under active development**. Expect changes in structure, functionality, and APIs in the near future.

## 📄 License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (the "License"). You may not use, copy, modify, or distribute this project except in compliance with the License. A copy of the License is included in the [LICENSE](./LICENSE) file in this repository.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations.

**Note**: This license applies to the source code in this repository. Trained models and datasets may be subject to separate licensing terms due to their dependencies or data sources. Please refer to the respective documentation for details.

## 👨‍💻 Author

Kourosh Sajjadi  
