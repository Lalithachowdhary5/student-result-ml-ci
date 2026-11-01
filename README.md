# Student Result Prediction ML CI

This project demonstrates how to add an ML quality gate to a GitHub Actions workflow.

The pipeline trains a student result prediction model, saves evaluation metrics to `metrics.json`, checks that the model achieves at least 85% accuracy, and only continues to the test stage when the gate passes.

## Repository structure

- `.github/workflows/ml-ci.yml` - CI workflow that runs the training and quality gate checks
- `train_model.py` - Creates a synthetic dataset, trains the model, and writes evaluation metrics
- `quality_gate.py` - Reads `metrics.json` and fails the workflow if accuracy is below the required threshold
- `test_ml_pipeline.py` - Verifies the training pipeline and quality gate behaviour
- `requirements.txt` - Python dependencies for the ML workflow

## Quality gate threshold

The workflow fails when the model accuracy is lower than `0.85`.

## Local execution

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python quality_gate.py
python -m unittest discover -v
```

## Expected result

The model should train successfully and achieve an accuracy of at least 0.85. When the gate passes, the GitHub Actions workflow continues to the automated tests.
