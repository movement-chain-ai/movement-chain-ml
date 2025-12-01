# Git Hooks & Development Workflow Setup

## Overview

This repository uses Git hooks to enforce code quality, testing standards, and conventional commit messages for the Movement Chain ML project. The setup combines Python tooling (Black, Ruff, isort, mypy, pytest) with Node.js-based commit validation (Husky, commitlint).

## Prerequisites

- **Python 3.11+** - Required for ML development
- **Node.js 20+** - Required for Git hooks and commit validation
- **Git** - Version control

## Installation

### 1. Install Node.js Dependencies

```bash
npm install
```

This will:
- Install Husky for Git hooks management
- Install commitlint for commit message validation
- Install lint-staged for running linters on staged files
- Automatically set up Git hooks via the `prepare` script

### 2. Install Python Development Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install production dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check Python tools
black --version
ruff --version
isort --version
mypy --version
pytest --version
nbstripout --version

# Check Node.js tools
npx commitlint --version
npx husky --version
```

### 4. Make Hooks Executable (if needed)

```bash
chmod +x .husky/commit-msg
chmod +x .husky/pre-commit
chmod +x .husky/pre-push
```

## Git Hooks

### commit-msg Hook

**Runs on:** Every commit
**Purpose:** Validates commit messages follow Conventional Commits format

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Valid Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `chore`: Maintenance tasks
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `style`: Code formatting
- `revert`: Revert previous commit
- `build`: Build system changes

**Examples:**
```bash
git commit -m "feat: add neural network model for prediction"
git commit -m "fix: resolve data loading issue in pipeline"
git commit -m "docs: update README with model architecture"
git commit -m "test: add unit tests for preprocessing module"
```

### pre-commit Hook

**Runs on:** Every commit
**Purpose:** Format and lint Python code, strip notebook outputs

**Actions:**
1. **Black** - Code formatting (auto-fix)
2. **Ruff** - Fast Python linter (auto-fix)
3. **isort** - Import sorting (auto-fix)
4. **nbstripout** - Remove notebook outputs

**What gets checked:**
- All staged `.py` files
- All staged `.ipynb` notebook files

**To bypass (not recommended):**
```bash
git commit --no-verify -m "feat: emergency fix"
```

### pre-push Hook

**Runs on:** Every push
**Purpose:** Run type checking and tests before pushing

**Actions:**
1. **mypy** - Type checking (non-blocking warnings)
2. **pytest** - Unit tests with coverage (blocking)

**What happens:**
- Type check all Python files in `src/`
- Run all tests in `tests/` directory
- Generate coverage report
- **Push will fail if tests fail**

**To bypass (not recommended):**
```bash
git push --no-verify
```

## Python Tools Configuration

### Black (Code Formatter)

**Configuration:** `pyproject.toml`
- Line length: 100 characters
- Target: Python 3.11

**Usage:**
```bash
# Format all files
black src/ tests/

# Check formatting without changes
black --check src/ tests/

# Format specific file
black src/models/model.py
```

### Ruff (Linter)

**Configuration:** `pyproject.toml`
- Line length: 100 characters
- Checks: pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade

**Usage:**
```bash
# Lint all files
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Lint specific file
ruff check src/models/model.py
```

### isort (Import Sorting)

**Configuration:** `pyproject.toml`
- Profile: black (compatible with Black)
- Line length: 100

**Usage:**
```bash
# Sort imports
isort src/ tests/

# Check without changes
isort --check-only src/ tests/
```

### mypy (Type Checking)

**Configuration:** `pyproject.toml`
- Target: Python 3.11
- Strict mode: Partially enabled

**Usage:**
```bash
# Type check all source files
mypy src/

# Type check specific file
mypy src/models/model.py
```

### pytest (Testing)

**Configuration:** `pyproject.toml`
- Test directory: `tests/`
- Coverage: Required for `src/`
- Coverage reports: terminal, HTML, XML

**Usage:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::test_training
```

### nbstripout (Notebook Output Stripping)

**Purpose:** Remove outputs from Jupyter notebooks before committing

**Usage:**
```bash
# Strip outputs from all notebooks
nbstripout notebooks/**/*.ipynb

# Dry run (check what would be stripped)
nbstripout --dry-run notebooks/**/*.ipynb

# Install as Git filter (alternative to pre-commit hook)
nbstripout --install
```

## CI/CD Pipeline

### PR Validation Workflow

**File:** `.github/workflows/pr-validation.yml`

**Triggers:**
- Pull request opened
- Pull request synchronized
- Pull request reopened

**Jobs:**

1. **validate-python**
   - Lint with Ruff
   - Format check with Black
   - Import order check with isort
   - Type check with mypy
   - Run tests with coverage
   - Upload coverage to Codecov

2. **validate-notebooks**
   - Check notebooks have outputs stripped
   - Validate notebook structure

3. **validate-commits**
   - Validate commit messages follow Conventional Commits

## Development Workflow

### Starting a New Feature

```bash
# 1. Create feature branch
git checkout -b feat/new-model

# 2. Make changes to Python files
vim src/models/new_model.py

# 3. Run linters manually (optional)
black src/
ruff check --fix src/
isort src/

# 4. Add and commit (hooks will run automatically)
git add src/models/new_model.py
git commit -m "feat: add new prediction model"

# 5. Push (tests will run automatically)
git push origin feat/new-model

# 6. Create pull request
# GitHub Actions will run full validation
```

### Working with Notebooks

```bash
# 1. Create or edit notebook
jupyter notebook notebooks/exploration.ipynb

# 2. When ready to commit, strip outputs
nbstripout notebooks/exploration.ipynb

# 3. Commit (pre-commit hook will verify)
git add notebooks/exploration.ipynb
git commit -m "feat: add data exploration notebook"
```

### Running Tests Locally

```bash
# Run all tests before pushing
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

## Troubleshooting

### Commit Message Rejected

**Error:** `❌ Commit message validation failed!`

**Solution:**
```bash
# Use correct format
git commit -m "feat: your feature description"

# Not like this
git commit -m "Added new feature"  # ❌ Missing type
git commit -m "Feat: New Feature"  # ❌ Capital letters
```

### Pre-commit Hook Fails

**Error:** Linting or formatting issues

**Solution:**
```bash
# Auto-fix most issues
black src/ tests/
ruff check --fix src/ tests/
isort src/ tests/

# Then retry commit
git add .
git commit -m "fix: resolve linting issues"
```

### Pre-push Hook Fails

**Error:** Tests failing

**Solution:**
```bash
# Run tests locally to debug
pytest -v

# Fix failing tests
vim tests/test_*.py

# Run specific failing test
pytest tests/test_model.py::test_specific_case -v

# Commit fixes and retry push
git add tests/
git commit -m "test: fix failing unit tests"
git push
```

### Husky Not Installed

**Error:** `husky - command not found`

**Solution:**
```bash
# Reinstall Node.js dependencies
npm install

# Manually initialize Husky
npx husky install
```

### Python Tools Not Found

**Error:** `black: command not found`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dev dependencies
pip install -r requirements-dev.txt
```

## Best Practices

### Code Quality
- Run linters before committing: `black . && ruff check --fix . && isort .`
- Keep line length under 100 characters
- Add type hints to function signatures
- Write docstrings for public functions and classes

### Testing
- Maintain test coverage above 80%
- Write tests before pushing code
- Use descriptive test names: `test_model_predicts_correct_output`
- Mock external dependencies

### Notebooks
- Always strip outputs before committing
- Keep notebooks focused on single topics
- Add markdown cells for documentation
- Avoid hardcoded paths (use environment variables)

### Commit Messages
- Use present tense: "add feature" not "added feature"
- Be specific: "fix data loading race condition" not "fix bug"
- Reference issues: "fix: resolve #123 - handle missing data"
- Keep subject line under 50 characters

### Git Workflow
- Create feature branches from `main`
- Keep PRs focused and small
- Request reviews from team members
- Squash commits before merging if needed

## Additional Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [pytest Documentation](https://docs.pytest.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Husky Documentation](https://typicode.github.io/husky/)

## Support

For issues or questions:
1. Check this documentation
2. Review CI/CD logs in GitHub Actions
3. Ask in team Slack channel
4. Open an issue in the repository
