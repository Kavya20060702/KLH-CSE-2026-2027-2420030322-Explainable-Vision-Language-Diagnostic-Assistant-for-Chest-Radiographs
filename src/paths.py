"""
Resolves models/ and outputs/ to the project root regardless of which
folder a script is run from.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../xray-assistant/src
PROJECT_ROOT = os.path.dirname(_THIS_DIR)  # .../xray-assistant

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(MODELS_DIR, "chest_classifier.pt")
THRESHOLDS_PATH = os.path.join(MODELS_DIR, "thresholds.json")
REPORT_INDEX_PATH = os.path.join(MODELS_DIR, "report_index.pkl")