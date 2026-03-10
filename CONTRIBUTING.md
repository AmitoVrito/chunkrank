# Contributing to ChunkRank

Thank you for your interest in contributing!

---

## Getting Started

```bash
git clone https://github.com/AmitoVrito/chunkrank.git
cd chunkrank
poetry install
```

---

## Development Workflow

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run tests before submitting
4. Open a pull request with a clear description of what and why

---

## Running Tests

```bash
poetry run pytest tests/
```

---

## Code Style

- Formatter: `ruff format`
- Linter: `ruff check`
- Type checker: `mypy`

Run all checks:
```bash
poetry run ruff format .
poetry run ruff check .
poetry run mypy chunkrank/
```

---

## Submitting Changes

- Keep commits focused and atomic
- Write clear commit messages
- Reference any related issues in your PR description

---

## Reporting Issues

Open an issue on GitHub: https://github.com/AmitoVrito/chunkrank/issues

Please include:
- What you expected to happen
- What actually happened
- A minimal code example to reproduce

---

## Getting Listed as a Contributor

After your PR is merged, open a follow-up PR adding your name to [CONTRIBUTORS.md](CONTRIBUTORS.md).
