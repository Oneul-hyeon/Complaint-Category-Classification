import os

# cuda version 11.6
os.system("pip install -r requirements.txt")
os.system("pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116")