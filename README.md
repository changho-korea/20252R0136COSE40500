# Titans Model Implementation

This repository contains the implementation of the **Titans** model from the paper ["Titans: Learning to Memorize at Test Time" (2501.00663)](https://arxiv.org/abs/2501.00663).

Implementations are available in both **PyTorch** and **JAX/Flax**.

## Prerequisites

-   Python 3.8+
-   pip

## Installation

1.  **Create a virtual environment** (recommended):

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: For JAX, you might need to install the specific version compatible with your accelerator (CUDA/TPU). See [JAX installation guide](https://github.com/google/jax#installation).*

## Usage

### PyTorch Implementation

-   **Model Code**: `titans.py`
-   **Demo**: `train_demo.py`

Run the PyTorch verification script:

```bash
python train_demo.py
```

### JAX/Flax Implementation

-   **Model Code**: `titans_jax.py`
-   **Demo**: `train_demo_jax.py`

Run the JAX verification script:

```bash
python train_demo_jax.py
```

## Structure

-   `read_pdf.py`: Script to extract text from the paper PDF.
-   `titans.py`: PyTorch implementation of Neural Memory and Titans MAC.
-   `titans_jax.py`: JAX/Flax implementation of Neural Memory and Titans MAC.
