# Contributing to Emotion Detection & Wellness Guidance System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

---

## 🤝 How to Contribute

### 1. **Report Bugs**
- Use GitHub Issues to report bugs
- Include:
  - Clear description
  - Steps to reproduce
  - Expected vs. actual behavior
  - Python version and OS
  - Error logs

### 2. **Suggest Enhancements**
- Open an issue with `[FEATURE]` prefix
- Describe the feature and its benefits
- Provide use cases

### 3. **Submit Code Changes**
- Fork the repository
- Create a feature branch: `git checkout -b feature/your-feature`
- Make your changes
- Write/update tests
- Commit with clear messages
- Push to your fork
- Open a Pull Request

---

## 📋 Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/aadrikash/emotion-ml.git
cd emotion-ml
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install pytest pytest-cov  # For testing
```

### 4. Create `.env` file
```bash
cp .env.example .env
# Edit .env with your settings
```

---

## 🎯 Code Standards

### Style Guide
- Follow PEP 8
- Use type hints where possible
- Write docstrings for all functions/classes
- Max line length: 100 characters

### Example Function:
```python
def predict_emotion(text: str, confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Predict emotional state from text.
    
    Args:
        text: Input text for analysis
        confidence_threshold: Minimum confidence for prediction
        
    Returns:
        Dictionary with emotion, intensity, and confidence
    """
    pass
```

### Naming Conventions
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

---

## 📝 Commit Message Format

Use clear, descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

### Types:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Code style (no logic change)
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Test additions/changes
- `chore:` - Build, dependencies, tooling

### Examples:
```bash
git commit -m "feat: Add BERT-based emotion classifier"
git commit -m "fix: Resolve NaN handling in preprocessing"
git commit -m "docs: Update README with new features"
```

---

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
pytest tests/ --cov=src  # With coverage
```

### Write Tests
Place tests in `tests/` directory:
```python
import pytest
from src.models import EmotionalStateModel

def test_emotion_model_initialization():
    model = EmotionalStateModel()
    assert model.model is None
    assert model.label_encoder is not None

def test_emotion_model_prediction():
    # Your test here
    pass
```

---

## 🔍 Pull Request Process

1. **Update Branch**
   ```bash
   git pull origin main
   git rebase main
   ```

2. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

3. **Create PR**
   - Clear title and description
   - Link related issues: `Closes #123`
   - Add labels (bug, enhancement, etc.)

4. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   
   ## Testing Done
   - [ ] Unit tests
   - [ ] Integration tests
   - [ ] Manual testing
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] No breaking changes
   ```

---

## 📚 Project Structure

```
emotion-ml/
├── src/                    # Source code
│   ├── preprocessing.py    # Data preprocessing
│   ├── models.py           # ML models
│   ├── decision_engine.py  # Decision logic
│   ├── uncertainty.py      # Confidence scoring
│   └── utils.py            # Utilities
├── tests/                  # Unit tests
├── data/                   # Datasets
├── models/                 # Trained models
├── results/                # Output results
├── notebooks/              # Jupyter notebooks
├── main.py                 # Main pipeline
├── config.py               # Configuration
└── requirements.txt        # Dependencies
```

---

## 🚀 Areas for Contribution

### High Priority
- [ ] Add deep learning models (LSTM, Transformers)
- [ ] Improve text preprocessing (NLP techniques)
- [ ] Add more emotional states
- [ ] Create web UI

### Medium Priority
- [ ] Add more recommendation actions
- [ ] Enhance decision engine rules
- [ ] Add data visualization dashboards
- [ ] Performance optimization

### Low Priority
- [ ] Documentation improvements
- [ ] Code refactoring
- [ ] Test coverage expansion
- [ ] GitHub Actions CI/CD

---

## 📞 Questions?

- Check existing issues and discussions
- Open a new discussion for questions
- Email: aadrika@example.com (if applicable)

---

## 📄 License

By contributing, you agree your contributions are licensed under the MIT License.

---

**Thank you for contributing! 🎉**