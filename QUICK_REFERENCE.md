# Quick Reference Card

## Setup (First Time)

```bash
# Automated setup
./setup.sh
source venv/bin/activate
pip install -r requirements-dev.txt

# Verify
black --version && ruff --version && pytest --version
```

## Daily Development

### Starting Work
```bash
cd /Users/maxwsy/Desktop/workspace/movement-chain-ml
source venv/bin/activate
git checkout -b feat/my-feature
```

### Making Changes
```bash
# Edit code in src/
vim src/models/my_model.py

# Edit tests
vim tests/test_my_model.py

# Run tests locally
pytest

# Format manually (optional - hooks will do this)
black src/ tests/
ruff check --fix src/ tests/
isort src/ tests/
```

### Committing
```bash
git add .
git commit -m "feat: add new model implementation"
# Hooks run automatically:
# ✅ commit-msg: Validates message format
# ✅ pre-commit: Formats code, strips notebooks
```

### Pushing
```bash
git push origin feat/my-feature
# Hooks run automatically:
# ✅ pre-push: Runs type checks and tests
```

## Commit Message Cheat Sheet

```bash
# Format: <type>(<scope>): <subject>

# Valid types:
feat      # New feature
fix       # Bug fix
docs      # Documentation
test      # Tests
refactor  # Code refactoring
perf      # Performance
chore     # Maintenance
ci        # CI/CD changes
style     # Code style
revert    # Revert commit
build     # Build system

# Examples:
git commit -m "feat: add neural network training"
git commit -m "fix: resolve data loading bug"
git commit -m "test: add unit tests for preprocessing"
git commit -m "docs: update API documentation"
git commit -m "refactor: simplify model architecture"
```

## Common Commands

### Python Tools
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Fix linting issues
ruff check --fix src/ tests/

# Sort imports
isort src/ tests/

# Type check
mypy src/

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

### Notebook Commands
```bash
# Strip outputs
nbstripout notebooks/*.ipynb

# Check if stripped (dry run)
nbstripout --dry-run notebooks/*.ipynb

# Start Jupyter
jupyter notebook
```

### Git Hook Commands
```bash
# Skip commit hooks (NOT RECOMMENDED)
git commit --no-verify -m "message"

# Skip push hooks (NOT RECOMMENDED)
git push --no-verify

# Reinstall hooks
npm install
npx husky install
```

## Troubleshooting

### Commit Message Rejected
```bash
# ❌ Wrong: "Added feature"
# ✅ Right: "feat: add feature"

# Format must be lowercase with type
git commit -m "feat: your description here"
```

### Pre-commit Fails
```bash
# Auto-fix most issues
black src/ tests/
ruff check --fix src/ tests/
isort src/ tests/

# Try commit again
git commit -m "fix: resolve formatting issues"
```

### Pre-push Fails (Tests)
```bash
# Run tests locally to debug
pytest -v

# Run specific test
pytest tests/test_model.py::test_function -v

# Fix and retry
git push
```

### Husky Not Working
```bash
# Reinstall
rm -rf node_modules package-lock.json
npm install
```

### Python Tools Not Found
```bash
# Activate venv
source venv/bin/activate

# Reinstall
pip install -r requirements-dev.txt
```

## File Locations

```
Configuration Files:
├── pyproject.toml          # Python tools config
├── package.json            # Node.js dependencies
├── .lintstagedrc.json     # Lint-staged config
├── commitlint.config.js   # Commit rules
├── .gitignore             # Git ignore
└── requirements-dev.txt   # Python dev deps

Git Hooks:
├── .husky/commit-msg      # Validates commits
├── .husky/pre-commit      # Formats code
└── .husky/pre-push        # Runs tests

CI/CD:
└── .github/workflows/pr-validation.yml

Documentation:
├── README.md              # Quick start
├── SETUP_INSTRUCTIONS.md  # Detailed setup
├── HOOKS_SETUP.md         # Complete guide
├── IMPLEMENTATION_SUMMARY.md  # Tech summary
└── QUICK_REFERENCE.md     # This file
```

## Tool Versions

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| Node.js | 20+ | Hooks |
| black | 24.0.0 | Formatter |
| ruff | 0.3.0 | Linter |
| isort | 5.13.0 | Import sorter |
| mypy | 1.8.0 | Type checker |
| pytest | 8.0.0 | Testing |
| husky | 9.0.0 | Git hooks |
| commitlint | 19.0.0 | Commit validation |

## Help

- **Setup issues**: See `SETUP_INSTRUCTIONS.md`
- **Workflow help**: See `HOOKS_SETUP.md`
- **Full details**: See `IMPLEMENTATION_SUMMARY.md`
- **Project info**: See `README.md`

## Quick Fixes

```bash
# Reset everything
rm -rf venv node_modules package-lock.json
./setup.sh

# Fix permissions
chmod +x .husky/* setup.sh

# Update tools
pip install --upgrade -r requirements-dev.txt
npm update

# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Black Style Guide](https://black.readthedocs.io/)
- [Ruff Rules](https://docs.astral.sh/ruff/rules/)
- [pytest Docs](https://docs.pytest.org/)
- [Husky Guide](https://typicode.github.io/husky/)
