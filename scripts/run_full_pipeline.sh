#!/bin/zsh
# To run each step of the pipeline one by one, follow these steps:

python src/cli.py validate --config configs/data.yaml

python src/cli.py features --experiment-config configs/experiment.yaml --data-config configs/data.yaml

python src/cli.py train --experiment-config configs/experiment.yaml --data-config configs/data.yaml

python src/cli.py evaluate --run-dir <run-dir>

python src/cli.py predict --run-dir rtifacts/20250814-164331/ --inference-config configs/inference.yaml

# Save to a custom path instead of inside run-dir
python src/cli.py submit \
    --predictions-path artifacts/20250814-164331/predictions.csv \
    --output-path artifacts/20250814-164331/my_submission.csv

# Submit directly to Kaggle after building
python src/cli.py submit \
    --predictions-path artifacts/20250812-184955/predictions.csv \
    --remote \
    --message "My best CV run so far"

# If your predictions.csv has only probabilities, force a threshold
python src/cli.py submit \
    --predictions-path artifacts/20250812-184955/proba_predictions.csv \
    --threshold 0.61
