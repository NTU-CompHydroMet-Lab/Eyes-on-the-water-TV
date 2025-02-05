import json
import os

BASELINE_FILE = 'baselines.json'

def create_default_config(config_path=BASELINE_FILE):
    """Create a default configuration file if it doesn't exist."""
    default_config = {
        "_template": {
            "description": "Segmentation baseline values for each parent/level1 folder combination",
            "format": {
                "parent_folder/folder_name": "baseline_value"
            }
        }
    }
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    return default_config

def load_baselines():
    # if not exists, create a default config
    if not os.path.exists(BASELINE_FILE):
        create_default_config()

    try: # load if the file exists and is valid
        with open(BASELINE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e: # Create a default config if the file is not valid
        create_default_config()
        with open(BASELINE_FILE, 'r') as f:
            return json.load(f)

def save_baselines(baselines):
    with open(BASELINE_FILE, 'w') as f:
        json.dump(baselines, f)

def get_baseline(parent_folder, folder):
    """Get baseline for a specific parent/folder combination."""
    baselines = load_baselines()
    key = f"{parent_folder}/{folder}"
    return baselines.get(key)

def set_baseline(parent_folder, folder, value):
    """Set baseline for a specific parent/folder combination."""
    baselines = load_baselines()
    key = f"{parent_folder}/{folder}"
    baselines[key] = value
    save_baselines(baselines)