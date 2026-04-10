# Release Guide (PyPI)

This project is configured for wheel + source distribution publishing.

## 1) Install release tooling

```bash
python -m pip install --upgrade build twine
```

## 2) Run tests

```bash
python -m pytest
```

## 3) Build distributions

```bash
python -m build
```

Artifacts are created in `dist/`:
- `ai_bridge_kit-<version>-py3-none-any.whl`
- `ai_bridge_kit-<version>.tar.gz`

## 4) Validate package metadata

```bash
python -m twine check dist/*
```

## 5) Upload to TestPyPI (recommended first)

Set credentials:

Windows PowerShell:

```powershell
$env:TWINE_USERNAME="__token__"
$env:TWINE_PASSWORD="pypi-<testpypi-token>"
```

Upload:

```bash
python -m twine upload --repository testpypi dist/*
```

## 6) Upload to PyPI

Set production token:

Windows PowerShell:

```powershell
$env:TWINE_USERNAME="__token__"
$env:TWINE_PASSWORD="pypi-<pypi-token>"
```

Upload:

```bash
python -m twine upload dist/*
```

## 7) Tag release in git

```bash
git tag v0.1.0
git push origin v0.1.0
```
