# Movement Chain ML

Machine Learning components for the Movement Chain project.

## Quick Start

### Installation

1. **Install Node.js dependencies** (for Git hooks):
   ```bash
   npm install
   ```

2. **Set up Python environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -r requirements-dev.txt

   # Install production dependencies (when available)
   # pip install -r requirements.txt
   ```

3. **Verify setup**:
   ```bash
   # Test Python tools
   black --version
   ruff --version
   pytest --version

   # Test Node tools
   npx husky --version
   npx commitlint --version
   ```

## Project Structure

```
movement-chain-ml/
├── src/                  # Source code
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks
├── .github/
│   └── workflows/        # GitHub Actions workflows
├── .husky/               # Git hooks
├── pyproject.toml        # Python tool configuration
├── requirements-dev.txt  # Development dependencies
└── HOOKS_SETUP.md       # Detailed Git hooks documentation
```

## Development Workflow

### Making Changes

```bash
# 1. Create feature branch
git checkout -b feat/your-feature

# 2. Make changes
# ... edit files ...

# 3. Format and lint (automatic on commit)
black src/ tests/
ruff check --fix src/ tests/
isort src/ tests/

# 4. Run tests
pytest

# 5. Commit with conventional format
git commit -m "feat: add new feature"

# 6. Push (tests run automatically)
git push origin feat/your-feature
```

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

**Types:** feat, fix, docs, chore, refactor, test, perf, ci, style

**Examples:**
```bash
git commit -m "feat: add model training pipeline"
git commit -m "fix: resolve data preprocessing issue"
git commit -m "docs: update README with examples"
git commit -m "test: add unit tests for feature extraction"
```

## Git Hooks

- **commit-msg**: Validates commit message format
- **pre-commit**: Runs linters and formatters, strips notebook outputs
- **pre-push**: Runs type checking and tests

See [HOOKS_SETUP.md](HOOKS_SETUP.md) for detailed documentation.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_model.py -v
```

## Code Quality Tools

- **Black**: Code formatting (line length: 100)
- **Ruff**: Fast Python linter
- **isort**: Import sorting
- **mypy**: Type checking
- **pytest**: Testing framework

## CI/CD

Pull requests automatically run:
- Python linting (Ruff, Black, isort)
- Type checking (mypy)
- Unit tests with coverage
- Notebook validation
- Commit message validation

## Documentation

- [HOOKS_SETUP.md](HOOKS_SETUP.md) - Complete Git hooks and development workflow guide

## License

[Add license information]

## Contributing

[Add contribution guidelines]
