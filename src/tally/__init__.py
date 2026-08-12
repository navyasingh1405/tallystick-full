"""Tallystick.

The governance layer is the product; the model is scaffolding.
"""

import os

# SHAP draws a tqdm progress bar on stderr for interventional explanations over more
# than a handful of rows. It interleaves badly with CLI output and with pytest
# capture, and this application never wants it. Set before shap is imported.
os.environ.setdefault("TQDM_DISABLE", "1")

__version__ = "0.1.0"
