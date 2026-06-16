# Train Very Small Model

This repository contains scripts to train a miniature, character-level transformer language model from scratch. It is designed to be small enough to train on a laptop (using Apple Silicon MPS or CPU). It's built similarly to a nanoGPT-style architecture and includes tools to prepare a dataset from any local code repository, train the model, and generate text from the trained weights.

## Project Structure

- `prep_data.py`: A utility script that compiles a local codebase into a single text file (`train.txt`) to be used as the training dataset.
- `train.py`: The main training loop. Defines the model architecture (a character-level Transformer), loads the dataset, trains the model, and saves the final weights to `mac_coder_model.pth`.
- `generate.py`: A helper script containing a function for loading the trained model weights (`mac_coder_model.pth`) and generating new text sequences.

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies. The required packages are specified in `pyproject.toml`.

1. Install `uv` if you haven't already.
2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

### 1. Prepare the Data

You need a dataset to train on. The `prep_data.py` script will compile files from a specified repository into `train.txt`.

First, create a `.env` file in the root directory and specify the path to the code you want to train on:

```env
PATH_TO_TRAINING_CODE=/path/to/your/local/repo
```

Then, run the data preparation script:

```bash
uv run prep_data.py
```
This will generate `train.txt` in the current directory.

### 2. Train the Model

Once your `train.txt` is ready, you can start training. The model is configured with hyperparameters suitable for a small model. By default, it will use `mps` (Metal Performance Shaders) if available on macOS, otherwise it will fall back to `cpu`.

```bash
uv run train.py
```

During training, the script will print the training and validation loss. At the end of the training iterations, it will output a sample of generated text and save the model weights to `mac_coder_model.pth`.

### 3. Generate Code

You can use the `load_previous_model_weights` function from `generate.py` to load the saved `mac_coder_model.pth` and generate text. Make sure to pass the appropriate model architecture and device.
