# Eyes-on-the-water-TV

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NTU-CompHydroMet-Lab/Eyes-on-the-water-TV
   cd Eyes-on-the-water-TV
   ```

2. Install the required packages:
   ```bash
   # You can either use the venv or conda environment
   python -m venv EOTWTV_env
   source EOTWTV_env/bin/activate   # On Windows use `EOTWTV_env\Scripts\activate`
   pip install uv
   uv pip install -r requirements.txt
   ```
   
   ```bash
   # Or create a conda environment
   conda env create -n eyes-on-the-water
   conda activate eyes-on-the-water
   pip install uv
   uv pip install -r requirements.txt
   ```

## Usage
To run the app, execute:
```bash
python EOTWTV_app.py
```

If there is an error about the port, you can change the port by:
```bash
# By default, the port is 8051
python EOTWTV_app.py --port XXXX # You can choose any port that is available, e.g. 8052
```

After the launch, you can see the app by opening the following link in your browser:
```
http://127.0.0.1:8051 # http://127.0.0.1:XXXX
```

Reminder: The "cache" folder is used to reduce computation. Feel free to delete it if the cache is not useful anymore.

![image](assets/demo.gif)

## Batch Processing
For batch processing details, please refer to the "batch_process.ipynb".

# Batch process
Please refer to the "batch_process.ipynb"



