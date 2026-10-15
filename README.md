# Train Very Small Model

This repository contains scripts to train a miniature transformer language model from scratch on source code. It is designed to be small enough to train on a laptop (using Apple Silicon MPS or CPU). It's built similarly to a nanoGPT-style architecture (4 layers, 4 heads, 128-dim embeddings, 128-token context) and uses the GPT-4 BPE tokenizer (`tiktoken` `cl100k_base`, ~100k vocab).

There are two workflows:

- **Simple training** (`train/`): compile a single Go repository into a text file and train on it.
- **Pretraining** (`pretrain/` + `train/train_bpe.py`): tokenize many repositories in parallel into binary files, then train on them using memory-mapped data.

## Project Structure

### `train/`

- `prep_data.py`: Compiles the `.go` files (excluding tests) from a local repository into a single text file (`train.txt`), with a `// File: <name>` header before each file.
- `train.py`: Defines the model architecture (a Transformer with multi-head causal self-attention), trains it on `train.txt`, prints a generated sample, and saves the weights to `mac_coder_model.pth`.
- `train_bpe.py`: Same model, but trains on the pre-tokenized `train.bin`/`val.bin` files produced by `pretrain/pretrain.py`.
- `generate.py`: Loads saved weights from `mac_coder_model.pth` and generates text.

### `pretrain/`

- `pretrain.py`: Scans every repository in a folder for source files (`.go`, `.py`, `.js`, `.ts`, `.rs`, `.cpp`, `.h`, `.java`), tokenizes them in parallel using all CPU cores, splits them 90/10 into train/validation, and writes `pretokenized_data/train.bin` and `pretokenized_data/val.bin` (`uint32` token IDs).

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies. `train/` and `pretrain/` are separate projects, each with its own `pyproject.toml`.

1. Install `uv` if you haven't already.
2. Install dependencies for each project:
   ```bash
   cd train && uv sync
   cd ../pretrain && uv sync
   ```

## Simple Training

All commands are run from the `train/` folder.

### 1. Prepare the Data

Create `train/.env` with the path to the repository you want to train on. Relative paths are resolved from the `train/` folder:

```env
PATH_TO_TRAINING_CODE=/path/to/your/local/repo
```

Then run:

```bash
uv run prep_data.py
```

This will generate `train.txt` in the `train/` folder.

### 2. Train the Model

```bash
uv run train.py
```

By default, it will use `mps` (Metal Performance Shaders) if available on macOS, otherwise it will fall back to `cpu`. During training, the script prints the training and validation loss. At the end, it prints a sample of generated text and saves the model weights to `mac_coder_model.pth`.

If `mac_coder_model.pth` already exists, training resumes from those weights.

## Pretraining

### 1. Tokenize the Repositories

Create `pretrain/.env` with the path to a folder containing one or more repositories. Every subfolder is treated as a repository. Relative paths are resolved from the `pretrain/` folder:

```env
PATH_TO_REPOS_FOLDER=/path/to/folder/of/repos
```

Then, from the `pretrain/` folder, run:

```bash
uv run pretrain.py
```

This writes `train.bin` and `val.bin` to `pretrain/pretokenized_data/`.

### 2. Train on the Tokenized Data

From the `train/` folder, run:

```bash
uv run train_bpe.py
```

This reads `../pretrain/pretokenized_data/*.bin` with `np.memmap`, so the dataset doesn't have to fit in memory. Like `train.py`, it resumes from `mac_coder_model.pth` if it exists and saves the weights there when finished. Both scripts use the same model and checkpoint file, so you can pretrain with `train_bpe.py` and then continue training on a single repository with `train.py`.

## Generate Code

From the `train/` folder, run:

```bash
uv run generate.py
```

This loads `mac_coder_model.pth` and prints 300 generated tokens. You can also use the `load_previous_model_weights(device, path)` function from your own code.

Values set in your shell environment take precedence over values in `.env` files.
