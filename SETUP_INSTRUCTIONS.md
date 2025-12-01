# Movement Chain ML - Setup Instructions

## Quick Setup

Run the automated setup script:

```bash
./setup.sh
```

Then activate the virtual environment and install Python dependencies:

```bash
# Activate virtual environment
source venv/bin/activate  # On Unix/macOS
# OR
venv\Scripts\activate     # On Windows

# Install Python dependencies
pip install -r requirements-dev.txt
```

## Manual Setup

If you prefer manual setup:

### 1. Install Node.js Dependencies

```bash
npm install
```

This installs:
- Husky (Git hooks manager)
- commitlint (Commit message validator)
- lint-staged (Run linters on staged files)

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Unix/macOS
# OR
venv\Scripts\activate     # Windows

# Install development tools
pip install -r requirements-dev.txt

# Install production dependencies (when available)
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Python tools
black --version
ruff --version
isort --version
mypy --version
pytest --version
nbstripout --version

# Node tools
npx husky --version
npx commitlint --version
```

### 4. Make Hooks Executable (Unix/macOS only)

```bash
chmod +x .husky/commit-msg
chmod +x .husky/pre-commit
chmod +x .husky/pre-push
```

## What Gets Installed

### Node.js Tools (package.json)

| Tool | Purpose | Version |
|------|---------|---------|
| husky | Git hooks manager | ^9.0.0 |
| @commitlint/cli | Commit message linter | ^19.0.0 |
| @commitlint/config-conventional | Conventional commits config | ^19.0.0 |
| lint-staged | Run linters on staged files | ^15.0.0 |

### Python Tools (requirements-dev.txt)

| Tool | Purpose | Version |
|------|---------|---------|
| black | Code formatter | 24.0.0 |
| ruff | Fast Python linter | 0.3.0 |
| isort | Import sorter | 5.13.0 |
| mypy | Type checker | 1.8.0 |
| pytest | Testing framework | 8.0.0 |
| pytest-cov | Coverage plugin | 4.1.0 |
| nbstripout | Notebook output stripper | 0.7.0 |
| nbconvert | Notebook converter | 7.14.0 |
| jupyter | Jupyter notebook | 1.0.0 |

## First Commit Test

Test that Git hooks are working:

```bash
# This should fail (bad commit message)
git commit --allow-empty -m "bad commit message"
# Expected: ❌ Commit message validation failed!

# This should succeed
git commit --allow-empty -m "chore: test git hooks setup"
# Expected: ✅ Commit message validated
```

## Troubleshooting

### Husky Not Working

```bash
# Reinstall and reinitialize
rm -rf node_modules package-lock.json
npm install
npx husky install
```

### Python Tools Not Found

```bash
# Make sure virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall if needed
pip install -r requirements-dev.txt --force-reinstall
```

### Hooks Not Executable

```bash
# Make them executable (Unix/macOS)
chmod +x .husky/*

# On Windows, no action needed
```

## Next Steps

1. **Read the documentation**: `HOOKS_SETUP.md` for complete workflow guide
2. **Try making a commit**: Test the pre-commit hooks
3. **Run tests**: `pytest` to verify everything works
4. **Start coding**: Create your first feature in `src/`

## File Structure

```
movement-chain-ml/
├── .github/
│   └── workflows/
│       └── pr-validation.yml    # CI/CD workflow
├── .husky/
│   ├── commit-msg               # Commit message validation
│   ├── pre-commit               # Linting and formatting
│   └── pre-push                 # Tests before push
├── src/
│   └── __init__.py              # Source code directory
├── tests/
│   ├── __init__.py
│   └── test_example.py          # Example tests
├── notebooks/                    # Jupyter notebooks (create as needed)
├── .gitignore                   # Git ignore rules
├── .lintstagedrc.json          # Lint-staged configuration
├── commitlint.config.js        # Commit message rules
├── package.json                 # Node.js dependencies
├── pyproject.toml              # Python tool configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── setup.sh                    # Automated setup script
├── HOOKS_SETUP.md              # Complete documentation
├── SETUP_INSTRUCTIONS.md       # This file
└── README.md                   # Project overview
```

## Support

For issues:
1. Check `HOOKS_SETUP.md` for detailed troubleshooting
2. Verify all tools are installed: Run verification commands above
3. Check Git hooks are executable: `ls -la .husky/`
4. Review CI/CD logs in GitHub Actions (after first push)

## Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [pytest Documentation](https://docs.pytest.org/)
- [Husky Documentation](https://typicode.github.io/husky/)
