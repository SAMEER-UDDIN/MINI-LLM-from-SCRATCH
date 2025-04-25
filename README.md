# Mini LLM from Scratch

This repository contains a simple implementation of a transformer-based language model (LLM) built from scratch using PyTorch.

## Features
- Tokenization using GPT-2's BPE tokenizer (`tiktoken`)
- Dataset preparation for next-token prediction
- Transformer Encoder architecture
- Training loop with CrossEntropyLoss

## Files
- `main.py`: Main model training script
- `train.txt`: Training text file (You need to add this manually)

## Usage
```bash
pip install -r requirements.txt
python main.py
```

## Note
Ensure you have a `train.txt` file in the same directory for this model to train properly.

## License
MIT License