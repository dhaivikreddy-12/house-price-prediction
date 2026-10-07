# Entry point: generate data, then train.
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "src/load_data.py"], check=True)
    subprocess.run([sys.executable, "src/train_model.py"], check=True)
