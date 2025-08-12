#!/bin/zsh
# To run each step of the pipeline one by one, follow these steps:

python src/cli.py validate --config configs/data.yaml

python src/cli.py features --experiment-config configs/experiment.yaml --data-config configs/data.yaml

python src/cli.py train --experiment-config configs/experiment.yaml --data-config configs/data.yaml

python src/cli.py evaluate --run-dir <run-dir>

python src/cli.py predict --run-dir <run-dir> --inference-config configs/inference.yamlpython src/cli.py submit --predictions-path <predictions-path> --remote -m "Submission message"