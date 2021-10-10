# emlens
[![Build Status](https://travis-ci.org/skojaku/emlens.svg?branch=main)](https://travis-ci.org/skojaku/emlens)
[![Unit Test & Deploy](https://github.com/skojaku/emlens/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/emlens/actions/workflows/main.yml)

A lightweight toolbox for analyzing embedding space

## Requirements
- Python 3.7 or 3.8 (may not work on 3.9)

## Doc

https://emlens.readthedocs.io/en/latest/

## Install

```bash
pip install emlens
```

`emlens` uses [faiss library](https://github.com/facebookresearch/faiss), which has two versions, `faiss-cpu` and `faiss-gpu`.
As the name stands, `faiss-gpu` can leverage GPUs, thereby faster if you have GPUs. `emlens` uses `faiss-cpu` by default to avoid unnecessary GPU-related troubles.
But, you can still leverage the GPUs (which is recommended if you have) by installing `faiss-gpu` by

*with conda*:
```bash
conda install -c conda-forge faiss-gpu
```

or *with pip*:
```
pip install faiss-gpu
```


## Maintenance

Code Linting:
```bash
conda install -y -c conda-forge pre-commit
pre-commit install
```

Docsctring: sphinx format

Test:
```bash
python -m unittest tests/simple_test.py
```
