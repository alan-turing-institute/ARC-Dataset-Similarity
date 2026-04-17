# ARC-Dataset-Similarity

Repo for ARC project on dataset similarity measures.

## Developer

### Installation

If you are running on a Linux HPC with CUDA GPUs, you will need to make sure you are
using Python 3.10:

```bash
uv venv --python 3.10.20
```

Install the project and initialise the virtual environment:

```bash
uv sync
source .venv/bin/activate
```

Set up pre-commits:

```bash
pre-commit install
```

### Testing

Run pytest:

```bash
pytest tests
```

## License

Distributed under the terms of the [MIT license](LICENSE).
