# RePhraser

**RePhraser** is a Python-based paraphrasing tool that allows users to generate alternative versions of input sentences while preserving their original meaning. The goal is to assist with writing, content rewording, and NLP experimentation.

## 🚀 Features

- Text rephrasing using pretrained and custom fine-tuned models
- Web interface for easy interaction (via `web.py`)
- Modular code structure for training and extending models
- Uses Hugging Face Transformers and Sentence Transformers

## 🧠 How It Works

At the core of RePhraser is a paraphrasing pipeline built on:

- [`ramsrigouthamg/t5_paraphraser`](https://huggingface.co/ramsrigouthamg/t5_paraphraser)
- [`tancperin/t5-paraphraser-quora`](https://huggingface.co/tancperin/t5-paraphraser-quora)
- [`tancperin/t5-paraphraser-bible`](https://huggingface.co/tancperin/t5-paraphraser-bible)

These models generate paraphrases and are optionally evaluated using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## 📚 Datasets

Custom models were trained on the following Kaggle datasets:

- [Quora Question Pairs Dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset)
- [Bible Paraphrase Dataset](https://www.kaggle.com/datasets/nileedixon/paired-bible-verses-for-semantic-similarity)

## 🗂️ Project Structure

```
.
├── rephraser.py         # Core paraphrasing logic
├── web.py               # Flask-based web interface
├── Trainers/            # Scripts for training models on Quora and Bible datasets
├── web/                 # Web template/static files
```

## 🔧 Installation

```bash
pip install torch transformers sentencepiece flask sentence-transformers

git clone https://github.com/tancperin/RePhraser.git
cd RePhraser

python web.py
```

## 👤 Contributors

- [@0uzkhan](https://github.com/0uzkhan)
- [@tancperin](https://github.com/tancperin)

## 🙏 Acknowledgements

Special thanks to **ramsrigouthamg** for providing the [T5 Paraphraser model](https://huggingface.co/ramsrigouthamg/t5_paraphraser), which was foundational to this project.

We also acknowledge the contributions of the team in training two custom models:

- `tancperin/t5-paraphraser-quora`: Fine-tuned on the [Quora Question Pairs Dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset)
- `tancperin/t5-paraphraser-bible`: Fine-tuned on the [Bible Paraphrase Dataset](https://www.kaggle.com/datasets/nileedixon/paired-bible-verses-for-semantic-similarity)
