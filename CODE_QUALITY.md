# Code Quality and Testing

## Overview

DreamWalk maintains high code quality standards through automated testing, linting, formatting, and security scanning.

## Test Coverage

**Current Coverage: 70%+ (Target: 95%)**

The project uses pytest for testing with comprehensive unit and integration tests.

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=services --cov-report=term-missing --cov-report=html

# Run specific test files
pytest tests/services/signal-processor/test_eeg_processor.py -v

# Run tests by marker
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests only
pytest tests/ -m slow          # Slow tests only

# Run tests with specific Python version
python -m pytest tests/
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── services/
│   ├── signal-processor/
│   │   └── test_eeg_processor.py
│   ├── neural-decoder/
│   │   └── test_decoder.py
│   ├── realtime-server/
│   │   └── test_server.py
│   ├── texture-generator/
│   ├── narrative-layer/
│   └── web-dashboard/
└── integration/
    └── test_end_to_end.py
```

### Writing Tests

**Unit Tests:**
- Test individual functions and methods
- Use mocks for external dependencies
- Fast execution (< 1 second per test)
- Mark with `@pytest.mark.unit`

**Integration Tests:**
- Test service interactions
- May require external services (Redis, etc.)
- Longer execution time
- Mark with `@pytest.mark.integration`

**Example Test:**
```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.unit
class TestEEGProcessor:
    def test_bandpass_filtering(self):
        """Test bandpass filtering of EEG signals"""
        # Test implementation
        assert True
```

## Code Quality Tools

### Linting

**flake8:**
- Checks Python code style and quality
- Configuration: `.flake8`
- Run: `flake8 services/`

**pylint:**
- Comprehensive code analysis
- Configuration: `.pylintrc`
- Run: `pylint services/`

### Formatting

**black:**
- Automatic code formatting
- Configuration: `pyproject.toml`
- Run: `black services/`
- Check: `black --check services/`

**isort:**
- Import sorting
- Configuration: `pyproject.toml`
- Run: `isort services/`
- Check: `isort --check-only services/`

### Type Checking

**mypy:**
- Static type checking
- Configuration: `pyproject.toml`
- Run: `mypy services/`

### Security Scanning

**bandit:**
- Security vulnerability scanning
- Run: `bandit -r services/ -ll`

**safety:**
- Dependency vulnerability checking
- Run: `safety check`

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

### Workflows

1. **CI Workflow** (`.github/workflows/ci.yml`):
   - Runs tests on Python 3.10 and 3.11
   - Code quality checks (flake8, pylint, black, isort)
   - Security scanning (bandit, safety)
   - Coverage reporting
   - Uploads coverage to Codecov

2. **Security Workflow** (`.github/workflows/security.yml`):
   - Weekly security scans
   - Bandit security scanning
   - Dependency vulnerability checks
   - pip-audit for package auditing

### Running CI Checks Locally

```bash
# Install dependencies
pip install pytest pytest-cov pytest-asyncio pytest-mock
pip install flake8 pylint black isort mypy bandit safety

# Run tests
pytest tests/ -v --cov=services --cov-report=term-missing

# Run linting
flake8 services/
pylint services/

# Run formatting checks
black --check services/
isort --check-only services/

# Run type checking
mypy services/

# Run security scans
bandit -r services/ -ll
safety check
```

## Code Quality Standards

### Code Style

- Follow PEP 8 style guide
- Maximum line length: 100 characters
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Avoid deep nesting (max 4 levels)

### Documentation

- Write clear docstrings for all functions and classes
- Use Google-style docstrings
- Document complex algorithms
- Include examples in docstrings
- Keep README files up to date

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors appropriately
- Never expose internal error details to clients
- Handle edge cases gracefully

### Performance

- Profile code before optimizing
- Use async/await for I/O operations
- Cache expensive computations
- Avoid unnecessary database queries
- Monitor performance metrics

## Pre-commit Hooks

**Recommended pre-commit hooks:**

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

**Example `.pre-commit-config.yaml`:**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
```

## Coverage Goals

- **Overall Coverage**: 95%+
- **Critical Paths**: 100%
- **New Code**: 100%
- **Legacy Code**: 70%+ (incremental improvement)

## Contributing

Before submitting a pull request:

1. **Run Tests**: Ensure all tests pass
   ```bash
   pytest tests/ -v
   ```

2. **Check Coverage**: Ensure coverage meets targets
   ```bash
   pytest tests/ --cov=services --cov-report=term-missing
   ```

3. **Run Linters**: Fix all linting issues
   ```bash
   flake8 services/
   pylint services/
   ```

4. **Format Code**: Ensure code is properly formatted
   ```bash
   black services/
   isort services/
   ```

5. **Type Check**: Run type checking
   ```bash
   mypy services/
   ```

6. **Security Scan**: Check for security issues
   ```bash
   bandit -r services/ -ll
   safety check
   ```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [black Documentation](https://black.readthedocs.io/)
- [flake8 Documentation](https://flake8.pycqa.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [bandit Documentation](https://bandit.readthedocs.io/)

