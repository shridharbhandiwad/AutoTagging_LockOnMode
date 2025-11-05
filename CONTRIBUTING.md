# Contributing to Airborne Track Behavior Tagging Application

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Coding Standards](#coding-standards)

## Code of Conduct

This project follows a code of conduct that we expect all contributors to uphold:

- Be respectful and inclusive
- Focus on constructive feedback
- Collaborate openly
- Report unacceptable behavior

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/airborne-track-tagger.git
   cd airborne-track-tagger
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

## Development Setup

### Prerequisites

- Python 3.8+
- C++ compiler (gcc/clang/MSVC)
- Git

### Installing Development Tools

```bash
# Code formatting
pip install black isort

# Linting
pip install flake8 pylint mypy

# Testing
pip install pytest pytest-cov pytest-qt

# Documentation
pip install sphinx sphinx-rtd-theme
```

### Build and Install

```bash
# Build C++ extensions
python setup.py build_ext --inplace

# Install in development mode
pip install -e .
```

## Making Changes

### Types of Contributions

We welcome:

- **Bug fixes**: Fix issues or incorrect behavior
- **New features**: Add new functionality
- **Documentation**: Improve docs, examples, tutorials
- **Tests**: Add or improve test coverage
- **Performance**: Optimize code
- **Refactoring**: Improve code structure

### Before You Start

1. Check existing issues and pull requests
2. Open an issue to discuss major changes
3. Ensure your idea aligns with project goals

### Development Workflow

1. **Write code**
   - Follow coding standards (see below)
   - Add docstrings to functions/classes
   - Include type hints where applicable

2. **Write tests**
   - Add unit tests for new functionality
   - Ensure all tests pass
   - Aim for >80% code coverage

3. **Update documentation**
   - Update README if needed
   - Add API docs for new functions
   - Include usage examples

4. **Format and lint**
   ```bash
   make format
   make lint
   ```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_parsers.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Writing Tests

Tests should:
- Be in `tests/` directory
- Use descriptive names: `test_<function>_<scenario>`
- Test normal cases, edge cases, and error cases
- Use fixtures for common setup
- Be independent and repeatable

Example:
```python
def test_binary_parser_with_simple_struct():
    """Test parsing simple struct from binary file."""
    # Arrange
    struct_def = create_test_struct()
    parser = BinaryParser(struct_def)
    
    # Act
    records = parser.parse_file('test_data.bin')
    
    # Assert
    assert len(records) == 10
    assert records[0]['field1'] == expected_value
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """Brief description of function.
    
    Longer description if needed, explaining behavior,
    algorithms, or important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When and why this is raised
    
    Example:
        >>> function_name(42, "test")
        True
    """
    pass
```

### Documentation Files

When updating:
- `README.md`: High-level overview and quick start
- `docs/USER_GUIDE.md`: Detailed usage instructions
- `docs/API.md`: API reference
- Code comments: Explain "why", not "what"

## Submitting Changes

### Pull Request Process

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```
   
   Commit message format:
   ```
   <type>: <subject>
   
   <body>
   
   <footer>
   ```
   
   Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Go to GitHub and create PR
   - Fill out PR template
   - Link related issues
   - Request review

### PR Checklist

Before submitting:

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts
- [ ] Commit messages are clear
- [ ] PR description explains changes

### Review Process

- Maintainers will review within 1 week
- Address feedback promptly
- Be open to suggestions
- Once approved, maintainers will merge

## Coding Standards

### Python Style

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/):

- Line length: 100 characters
- Indentation: 4 spaces
- Imports: standard lib → third-party → local
- Use meaningful variable names
- Add type hints

```python
# Good
def calculate_speed(distance: float, time: float) -> float:
    """Calculate speed from distance and time."""
    return distance / time

# Bad
def calc(d, t):
    return d/t
```

### C++ Style

- Follow modern C++ practices (C++17)
- Use RAII for resource management
- Const correctness
- Clear naming conventions
- Document public APIs

### File Organization

```
module_name/
├── __init__.py       # Module exports
├── core.py           # Core functionality
├── utils.py          # Utility functions
└── tests/
    └── test_core.py  # Tests
```

### Naming Conventions

- **Classes**: `PascalCase`
- **Functions/methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

## Adding New Features

### New File Parser

1. Create parser in `parsers/`:
   ```python
   class CustomParser:
       def parse_file(self, filepath: str) -> pd.DataFrame:
           # Implementation
           pass
   ```

2. Update `FileDetector` to recognize format
3. Add tests in `tests/test_parsers.py`
4. Update documentation

### New ML Model

1. Inherit from `BaseModel` in `ml/models.py`
2. Implement required methods: `fit`, `predict`, `save`, `load`
3. Add to `ModelManager` widget
4. Add tests
5. Update training script

### New Behavior Tag

1. Update `ModelTrainer.TAG_DEFINITIONS`
2. Implement labeling logic in `_generate_labels()`
3. Retrain models
4. Update documentation

## Questions?

- Open an issue for questions
- Join discussions in GitHub Discussions
- Check existing documentation

## Thank You!

Your contributions make this project better for everyone!

---

**Happy Coding!** 🚀
