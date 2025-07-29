# Lacium Flower

**Lacium Flower** is a phonologically-aware Transformer-based model designed for learning representations sensitive to phonological structure in Brazilian Portuguese.

### Why the name *Lacium Flower*?

The name is inspired by a poem of Olavo Bilac, a symbolist poet from Brazil, who referred to the Portuguese language as the “última flor do Lácio” — “the last flower of Latium”. This tribute reflects our goal of preserving and enhancing the richness of phonological structures in a language that carries deep cultural and historical roots.

## Features

- Contrastive self-supervised objective based on phonological distance
- Multitask supervised pretraining on auxiliary phonological tasks
- Built on top of BERT encoder backbone

## Directory Structure

- `models/`: Model definition including encoder and heads
- `trainers/`: Training scripts for contrastive and multitask pretraining
- `utils/`: Loss functions and helper utilities

## Installation

```bash
pip install torch transformers
```

## Running the Model

```bash
python main.py
```

Note: This is a minimal implementation template and needs datasets and training logic to be fully functional.
