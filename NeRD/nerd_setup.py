"""Add NeRD-Rain source paths so model/utils can be imported from wrapper scripts."""
import os
import sys
NERD_DIR = os.path.dirname(os.path.abspath(__file__))
WARMUP_DIR = os.path.join(NERD_DIR, 'pytorch-gradual-warmup-lr')
def setup_paths():
    for path in (NERD_DIR, WARMUP_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)
