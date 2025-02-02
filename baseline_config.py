import json
import os

BASELINE_FILE = 'baselines.json'

def create_default_config(config_path=BASELINE_FILE):
    """Create a default configuration file if it doesn't exist."""
    if not os.path.exists(config_path):
        default_config = {
            "_template": {
                "description": "Segmentation baseline values for each level 1 folder",
                "format": {
                    "folder_name": "baseline_value"
                }
            }
        }
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        return default_config
    return None

def load_baselines():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_baselines(baselines):
    with open(BASELINE_FILE, 'w') as f:
        json.dump(baselines, f)

def get_baseline(folder):
    baselines = load_baselines()
    return baselines.get(folder)

def set_baseline(folder, value):
    baselines = load_baselines()
    baselines[folder] = value
    save_baselines(baselines) 