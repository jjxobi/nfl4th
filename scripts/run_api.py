# ruff: noqa: I001
from pathlib import Path

import nfl4th  # noqa: F401  (must import before uvicorn, see nfl4th/__init__.py for why)
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(Path(__file__).resolve().parent.parent),
    )
