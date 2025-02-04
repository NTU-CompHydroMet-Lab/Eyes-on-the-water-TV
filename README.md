# Eyes-on-the-water-TV

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/NTU-CompHydroMet-Lab/Eyes-on-the-water-TV
   cd Eyes-on-the-water-TV

2. Install the required packages:
   ```bash
   # You can either use the venv or conda environment
   python -m venv EOTWTV_env
   source EOTWTV_env/bin/activate   # On Windows use `EOTWTV_env\Scripts\activate`
   pip install uv
   uv pip install -r requirements.txt
   
   # Or create a conda environment
   conda env create -n eyes-on-the-water
   conda activate eyes-on-the-water
   pip install uv
   uv pip install -r requirements.txt

# Run the app

```bash
python EOTWTV_app.py
```
![image](assets/demo.gif)


# Batch process
Please refer to the "batch_process.ipynb"
