# emlens
A lightweight toolbox for analyzing embedding space

## Doc

https://emlens.readthedocs.io/en/latest/

## Install

With conda: (change `myenv` to your env name)
```
conda env update -n myenv --file environment.yml
```

With pip:
```
pip install -r requirements.txt .
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
