# emlens
[![Build Status](https://travis-ci.org/skojaku/emlens.svg?branch=main)](https://travis-ci.org/skojaku/emlens)
[![Unit Test & Deploy](https://github.com/skojaku/emlens/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/emlens/actions/workflows/main.yml)

A lightweight toolbox for analyzing embedding space

## Requirements
- Python 3.7 or 3.8 (may not work on 3.9)

## Doc

https://emlens.readthedocs.io/en/latest/

## Install

### Prerequisite

*conda*:
```bash
conda install -c conda-forge faiss-gpu
```

Or *pip*:
```
pip install faiss-gpu
```

*If faiss-gpu cannot be installed, use faiss-cpu instead.*

### Installing emlens

```bash
pip install emlens
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
