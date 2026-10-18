import os
import sys

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, CODE_DIR)

from app import app
