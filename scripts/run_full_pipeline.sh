#!/bin/zsh
# Titanic ML pipeline: preprocess, train, predict, and submit in one command

set -e

# Paths
TRAIN_RAW="data/train.csv"
TEST_RAW="data/test.csv"
TRAIN_PROCESSED="data/processed_train.csv"
TEST_PROCESSED="data/processed_test.csv"
PREPROCESSOR="artifacts/preprocessor.joblib"
SUBMISSION="artifacts/submission.csv"
CONFIG_DIR="config"
TARGET="Survived"

# 1. Preprocess training data (with RandomForest Age imputation)
echo "[1/5] Preprocessing training data..."
python src/ktl/features/preprocess.py preprocess \
  --input-path "$TRAIN_RAW" \
  --output-path "$TRAIN_PROCESSED" \
  --target "$TARGET" \
  --age-rf-impute True \
  --age-rf-features "Pclass,Sex,SibSp,Parch,Fare"

# 2. Preprocess test data using the saved pipeline
echo "[2/5] Preprocessing test data..."
python src/ktl/features/preprocess.py preprocess_test \
  --test-path "$TEST_RAW" \
  --pipeline-path "$PREPROCESSOR" \
  --output-path "$TEST_PROCESSED"

# 3. Train the model
echo "[3/5] Training model..."
python src/ktl/cli.py train \
  --train-csv "$TRAIN_PROCESSED" \
  --target "$TARGET" \
  --config-dir "$CONFIG_DIR"

# 4. Predict on test set
echo "[4/5] Generating submission..."
python src/ktl/cli.py predict \
  --test-csv "$TEST_PROCESSED" \
  --out-path "$SUBMISSION"

# 5. Submit to Kaggle
echo "[5/5] Submitting to Kaggle..."
python src/ktl/cli.py submit \
  --file "$SUBMISSION" \
  --message "Auto submission from pipeline script"

echo "Pipeline complete!"
