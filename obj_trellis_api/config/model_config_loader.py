import yaml
from pathlib import Path

class ModelConfig:
    _instance = None
    _data = None

    def __new__(cls, path: str = "model_config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            base_dir = Path(__file__).resolve().parent
            cls._path = (base_dir / path).resolve()
            cls._load()
        return cls._instance

    @classmethod
    def _load(cls):
        with open(cls._path, "r") as f:
            cls._data = yaml.safe_load(f)

    def get(self, section: str):
        if section not in self._data:
            raise KeyError(f"Section '{section}' not found in {self._path}")
        return self._data[section]

    def reload(self):
        """Optionally reload config at runtime."""
        self._load()
