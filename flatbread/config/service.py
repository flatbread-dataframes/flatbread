from typing import Any
from pathlib import Path
import json


class ConfigService:
    def __init__(self):
        self._config: dict[str, Any] | None = None
        self._sources: list[str] = []

    def __repr__(self) -> str:
        return f"ConfigService(loaded={self._config is not None})"

    def __str__(self) -> str:
        self._ensure_loaded()  # Force loading for inspection

        sources_info = []
        for i, source in enumerate(self._sources, 1):
            name = Path(source).name
            sources_info.append(f"  {i}. {name}")

        sources_str = "\n".join(sources_info) if sources_info else "  (none)"

        return f"ConfigService loaded from {len(self._sources)} sources:\n{sources_str}\n\nFinal config: {self._config}"

    def __getitem__(self, key):
        return self.config[key]

    def get(self, key, default=None):
        return self.config.get(key, default)

    @property
    def config(self) -> dict[str, Any]:
        self._ensure_loaded()
        return self._config

    @property
    def sources(self) -> list[str]:
        return self._sources.copy()

    def reload(self) -> None:
        self._config = None
        self._sources = []

    def update_runtime(self, updates: dict[str, Any]) -> None:
        self._ensure_loaded()
        self._config = deep_merge(
            self._config, # type: ignore
            updates,
        )

    def _load_config(self) -> None:
        self._sources.clear()

        # Start with defaults
        defaults_path = Path(__file__).parent / "config.defaults.json"
        config = json.loads(defaults_path.read_text())
        self._sources.append(str(defaults_path))

        # User config
        user_path = Path('~/.flatbread.json').expanduser()
        if user_path.exists():
            user_config = json.loads(user_path.read_text())
            config = deep_merge(config, user_config)
            self._sources.append(str(user_path))

        # Project config
        if project_path := find_project_config():
            project_config = json.loads(project_path.read_text())
            config = deep_merge(config, project_config)
            self._sources.append(str(project_path))

        self._config = config

    def _ensure_loaded(self) -> None:
        if self._config is None:
            self._load_config()


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict:
    """
    Deep merge two dictionaries, preserving structure from both.

    - When both dicts have a dict at the same key, merge recursively
    - When update has keys not in base, add them
    - When types mismatch (e.g., dict in one, value in other), prefer update

    Parameters
    ----------
    base : dict
        Base dictionary to merge into
    update : dict
        Dictionary with values to update base with

    Returns
    -------
    dict
        Merged dictionary
    """
    merged = base.copy()
    for key, update_val in update.items():
        if key not in merged:
            # Add keys from update that don't exist in base
            merged[key] = update_val
        elif isinstance(update_val, dict) and isinstance(merged[key], dict):
            # Recursively merge nested dictionaries
            merged[key] = deep_merge(merged[key], update_val)
        else:
            # Override base value with update value
            merged[key] = update_val
    return merged


def find_project_config(max_levels: int = 5) -> Path|None:
    """
    Find project-level .flatbread.json, traversing up from current directory.

    Parameters
    ----------
    max_levels : int, default 5
        Maximum number of directory levels to traverse upward

    Returns
    -------
    Path or None
        Path to the config file if found, None otherwise
    """
    current_dir = Path.cwd()
    home_dir = Path.home()

    # Check current directory and up to max_levels parent directories
    for _ in range(max_levels + 1):
        config_path = current_dir / ".flatbread.json"
        if config_path.is_file():
            return config_path

        # Stop if we've reached the filesystem root or home directory
        if current_dir == current_dir.parent or current_dir == home_dir:
            break

        # Move up one directory
        current_dir = current_dir.parent

    return None
