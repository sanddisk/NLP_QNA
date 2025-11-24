import os

# Force default language to English for app2
os.environ["APP_DEFAULT_LANG"] = "en"

import app  # noqa: F401


