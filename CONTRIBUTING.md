# Contributing

1. Fork and create a branch.
2. `pip install -e .[dev]`
3. Run `python -m unittest discover -s tests -v` and `python benchmarks/prueba.py` (emulated mode) before opening a PR.
4. Keep the project dependency-free (standard library only; `numpy` stays optional).
5. Never include real memories, `.env` files or `*.db` files in a PR.
