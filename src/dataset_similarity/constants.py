from pathlib import Path

PROJECT_DIR = (Path(__file__) / ".." / ".." / "..").resolve()
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_EMBEDDING_DIR = PROJECT_DIR / "embeddings"
