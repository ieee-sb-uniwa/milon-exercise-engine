import os
import yaml
from typing import Optional


# Directory where exercise YAML configs live (relative to this file's package)
_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def load_config(exercise_name: str, config_dir: Optional[str] = None) -> dict:
    """Load an exercise config YAML by exercise name.

    Args:
        exercise_name: Case-insensitive name that matches a YAML filename,
                       e.g. "squat", "pushup", "legraise".
        config_dir:    Override the default configs directory path.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If no matching config file is found.
    """
    search_dir = config_dir or _CONFIGS_DIR
    filename = f"{exercise_name.lower()}.yaml"
    path = os.path.normpath(os.path.join(search_dir, filename))

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No config file found for exercise '{exercise_name}' (looked for: {path})"
        )

    with open(path, "r") as f:
        return yaml.safe_load(f)
