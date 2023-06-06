import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.append(path)


CURR_DIR = os.path.dirname(__file__)
PROJ_ROOT = os.path.abspath(os.path.join(CURR_DIR, "../"))
add_path(PROJ_ROOT)
