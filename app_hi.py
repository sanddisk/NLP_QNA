import os

# Force default language to Hindi for app2
os.environ["APP_DEFAULT_LANG"] = "hi"

import app  # noqa: F401


