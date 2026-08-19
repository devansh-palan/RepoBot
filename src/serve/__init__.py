"""The FastAPI service and its single-page frontend.

The module is `api.py`, not `app.py`: re-exporting the `app` instance from a
module of the same name makes `src.serve.app` mean two different things
depending on how it is reached, which breaks attribute-path patching and reads
as a bug when it happens.
"""

from .api import app
from .metrics import RequestTimer

__all__ = ["RequestTimer", "app"]
