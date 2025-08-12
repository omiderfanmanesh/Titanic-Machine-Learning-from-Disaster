# Titanic ML Pipeline - Documentation

Welcome to the Titanic ML Pipeline documentation! This directory contains all project documentation, guides, and technical specifications.

## 📁 Documentation Structure

### Core Documentation
- **[README.md](../README.md)** - Main project overview and quick start guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed system architecture and design patterns
- **[RESTRUCTURING_COMPLETE.md](RESTRUCTURING_COMPLETE.md)** - Complete restructuring summary
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes

### Development & Technical
- **[REFACTOR_SUMMARY.md](REFACTOR_SUMMARY.md)** - Refactoring process and decisions
- **[AGENTS.md](AGENTS.md)** - AI agent specifications and workflows
- **[API.md](API.md)** - API reference and usage examples *(coming soon)*
- **[TESTING.md](TESTING.md)** - Testing strategy and guidelines *(coming soon)*

### User Guides
- **[QUICK_START.md](QUICK_START.md)** - Get started in 5 minutes *(coming soon)*
- **[CONFIGURATION.md](CONFIGURATION.md)** - Configuration guide *(coming soon)*
- **[EXAMPLES.md](EXAMPLES.md)** - Usage examples and tutorials *(coming soon)*

## 🚀 Quick Navigation

### For Users
1. Start with [README.md](../README.md) for project overview
2. Follow [QUICK_START.md](QUICK_START.md) for immediate setup
3. Check [EXAMPLES.md](EXAMPLES.md) for common use cases

### For Developers
1. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. Read [REFACTOR_SUMMARY.md](REFACTOR_SUMMARY.md) for implementation details
3. Consult [API.md](API.md) for development reference

### For Contributors
1. Check [CHANGELOG.md](CHANGELOG.md) for recent changes
2. Review [TESTING.md](TESTING.md) for testing requirements
3. Follow [AGENTS.md](AGENTS.md) for AI-assisted development

## 🛠️ Pipeline Features

This professional ML pipeline includes:

- **SOLID Architecture**: 15+ interfaces, dependency inversion, factory patterns
- **Complete ML Workflow**: Data → Features → Training → Evaluation → Submission
- **Multiple Data Sources**: CSV, Kaggle API, cached loading
- **Advanced Features**: 8+ transformations, ensemble methods, TTA
- **6+ Model Types**: Logistic, RF, GBM, SVM, XGBoost, CatBoost
- **Comprehensive Testing**: 100+ test cases, integration & unit tests
- **Professional CLI**: 8 commands for complete workflow
- **Production Ready**: Logging, monitoring, error handling, validation

## 📊 Project Structure

```
src/
├── cli.py                  # Command-line interface
├── core/                   # Interfaces & utilities
├── data/                   # Loading & validation
├── features/               # Engineering & transforms
├── modeling/               # Training & registry
├── cv/                     # Cross-validation
├── eval/                   # Evaluation metrics
├── infer/                  # Prediction & ensemble
└── submit/                 # Kaggle submissions
```

## 🎯 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# View available commands
python src/cli.py --help

# Get pipeline info
python src/cli.py info

# Download competition data
python src/cli.py download

# Train a model
python src/cli.py train --config configs/experiment.yaml
```

---

**💡 Tip**: Each documentation file includes detailed examples and code snippets for hands-on learning.

**🔗 Links**: [GitHub Repository](https://github.com/omiderfanmanesh/Titanic-Machine-Learning-from-Disaster) | [Kaggle Competition](https://www.kaggle.com/c/titanic)
