import importlib

# Resolve each export on first use. Importing them eagerly pulls
# pytorch_lightning, hydra and rich into every inference process, which costs
# several seconds of startup for code only the training entry points run.
_EXPORTS = {
    "braceexpand": ".braceexpand",
    "autocast_exclude_mps": ".context",
    "get_latest_checkpoint": ".file",
    "instantiate_callbacks": ".instantiators",
    "instantiate_loggers": ".instantiators",
    "RankedLogger": ".logger",
    "log_hyperparameters": ".logging_utils",
    "enforce_tags": ".rich_utils",
    "print_config_tree": ".rich_utils",
    "set_seed": ".seed",
    "extras": ".utils",
    "get_metric_value": ".utils",
    "task_wrapper": ".utils",
}

__all__ = [
    "enforce_tags",
    "extras",
    "get_metric_value",
    "RankedLogger",
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "print_config_tree",
    "task_wrapper",
    "braceexpand",
    "get_latest_checkpoint",
    "autocast_exclude_mps",
    "set_seed",
]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)
