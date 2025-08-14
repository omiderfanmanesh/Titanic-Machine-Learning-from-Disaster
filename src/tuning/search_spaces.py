"""Search space definitions for hyperparameter tuning."""

from typing import Dict, Any


class SearchSpaceFactory:
    """Factory for creating hyperparameter search spaces."""

    @staticmethod
    def get_search_space(model_name: str) -> Dict[str, Dict[str, Any]]:
        """Get search space configuration for a model."""
        search_spaces = {
            "logistic": {
                "C": {
                    "type": "float",
                    "low": 0.001,
                    "high": 100.0,
                    "log": True
                },
                "max_iter": {
                    "type": "int",
                    "low": 100,
                    "high": 1000
                }
                # Note: We'll handle penalty+solver combinations in a special way
            },

            "random_forest": {
                "n_estimators": {
                    "type": "int",
                    "low": 50,
                    "high": 500
                },
                "max_depth": {
                    "type": "int",
                    "low": 3,
                    "high": 20
                },
                "min_samples_split": {
                    "type": "int",
                    "low": 2,
                    "high": 20
                },
                "min_samples_leaf": {
                    "type": "int",
                    "low": 1,
                    "high": 10
                },
                "max_features": {
                    "type": "categorical",
                    "choices": ["sqrt", "log2", "auto"]
                },
                "bootstrap": {
                    "type": "categorical",
                    "choices": [True, False]
                }
            },

            "gradient_boosting": {
                "n_estimators": {
                    "type": "int",
                    "low": 50,
                    "high": 300
                },
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": False
                },
                "max_depth": {
                    "type": "int",
                    "low": 3,
                    "high": 10
                },
                "min_samples_split": {
                    "type": "int",
                    "low": 2,
                    "high": 20
                },
                "min_samples_leaf": {
                    "type": "int",
                    "low": 1,
                    "high": 10
                },
                "subsample": {
                    "type": "float",
                    "low": 0.6,
                    "high": 1.0
                }
            },

            "catboost": {
                "iterations": {
                    "type": "int",
                    "low": 100,
                    "high": 1000
                },
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": True
                },
                "depth": {
                    "type": "int",
                    "low": 4,
                    "high": 10
                },
                "l2_leaf_reg": {
                    "type": "float",
                    "low": 1.0,
                    "high": 10.0,
                    "log": True
                },
                "border_count": {
                    "type": "int",
                    "low": 32,
                    "high": 255
                },
                "bagging_temperature": {
                    "type": "float",
                    "low": 0.0,
                    "high": 1.0
                }
            },

            "xgboost": {
                "n_estimators": {
                    "type": "int",
                    "low": 50,
                    "high": 500
                },
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.3,
                    "log": True
                },
                "max_depth": {
                    "type": "int",
                    "low": 3,
                    "high": 10
                },
                "min_child_weight": {
                    "type": "int",
                    "low": 1,
                    "high": 10
                },
                "subsample": {
                    "type": "float",
                    "low": 0.6,
                    "high": 1.0
                },
                "colsample_bytree": {
                    "type": "float",
                    "low": 0.6,
                    "high": 1.0
                },
                "reg_alpha": {
                    "type": "float",
                    "low": 0.0,
                    "high": 1.0
                },
                "reg_lambda": {
                    "type": "float",
                    "low": 0.0,
                    "high": 1.0
                }
            },

            "svm": {
                "C": {
                    "type": "float",
                    "low": 0.001,
                    "high": 100.0,
                    "log": True
                },
                "kernel": {
                    "type": "categorical",
                    "choices": ["rbf", "poly", "sigmoid"]
                },
                "gamma": {
                    "type": "categorical",
                    "choices": ["scale", "auto"]
                }
            },

            "ridge": {
                "alpha": {
                    "type": "float",
                    "low": 0.001,
                    "high": 100.0,
                    "log": True
                },
                "solver": {
                    "type": "categorical",
                    "choices": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
                }
            }
        }

        if model_name not in search_spaces:
            raise ValueError(f"Search space not defined for model: {model_name}")

        return search_spaces[model_name]

    @staticmethod
    def sample_logistic_params(trial):
        """Sample compatible logistic regression parameters."""
        # Sample C and max_iter normally
        params = {
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1000)
        }

        # Sample penalty+solver combinations that are compatible
        penalty_solver_combo = trial.suggest_categorical(
            "penalty_solver_combo",
            ["l1_liblinear", "l1_saga", "l2_liblinear", "l2_saga", "elasticnet_saga"]
        )

        if penalty_solver_combo == "l1_liblinear":
            params["penalty"] = "l1"
            params["solver"] = "liblinear"
        elif penalty_solver_combo == "l1_saga":
            params["penalty"] = "l1"
            params["solver"] = "saga"
        elif penalty_solver_combo == "l2_liblinear":
            params["penalty"] = "l2"
            params["solver"] = "liblinear"
        elif penalty_solver_combo == "l2_saga":
            params["penalty"] = "l2"
            params["solver"] = "saga"
        elif penalty_solver_combo == "elasticnet_saga":
            params["penalty"] = "elasticnet"
            params["solver"] = "saga"
            # Add l1_ratio for elasticnet
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        return params

    @staticmethod
    def get_available_models() -> list:
        """Get list of models with defined search spaces."""
        return [
            "logistic", "random_forest", "gradient_boosting",
            "catboost", "xgboost", "svm", "ridge"
        ]
