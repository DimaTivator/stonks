import time
import subprocess

def run_scraper():
    subprocess.run(["python3", "data_pipeline.py"], check=True)

if __name__ == "__main__":
    while True:
        time.sleep(6 * 60 * 60)
        run_scraper()
