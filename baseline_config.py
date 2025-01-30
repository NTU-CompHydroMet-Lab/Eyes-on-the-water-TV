import json
import os

DEFAULT_CONFIG_PATH = 'segmentation_baselines.json'

def create_default_config(config_path=DEFAULT_CONFIG_PATH):
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

def load_baselines(config_path=DEFAULT_CONFIG_PATH):
    # Create config file if it doesn't exist
    create_default_config(config_path)
    
    with open(config_path, 'r') as f:
        baselines = json.load(f)
    
    # Remove template from returned data
    if '_template' in baselines:
        baselines = {k: v for k, v in baselines.items() if k != '_template'}
    return baselines

def save_baselines(baselines, config_path=DEFAULT_CONFIG_PATH):
    # Preserve template if it exists
    existing_data = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing_data = json.load(f)
    
    # Keep template if it exists
    if '_template' in existing_data:
        baselines = {'_template': existing_data['_template'], **baselines}
    
    with open(config_path, 'w') as f:
        json.dump(baselines, f, indent=4)

def get_baseline(level1_folder, baselines=None):
    if baselines is None:
        baselines = load_baselines()
    return baselines.get(level1_folder)

def set_baseline(level1_folder, value, config_path=DEFAULT_CONFIG_PATH):
    baselines = load_baselines(config_path)
    baselines[level1_folder] = value
    save_baselines(baselines, config_path) 