# run_all.py: runs the whole pipeline in order. Run after src/download_data.py.
# Usage (from the project root):  python src/run_all.py
import os
import subprocess
import sys
import time

STEPS = ["explore_data.py", "run_official.py", "run_cv.py", "analyze_errors.py", "accuracy_ci.py", "stat_tests.py", "make_summary.py"]
MARKER = "results/.pipeline_complete"


def main():
    assert os.path.exists("data/raw/uci_har"), "Dataset not found: run  python src/download_data.py  first"
    os.makedirs("results", exist_ok=True)
    if os.path.exists(MARKER):
        os.remove(MARKER)
    t0 = time.time()
    for step in STEPS:
        print(f"\n######## {step} ########", flush=True)
        t = time.time()
        subprocess.run([sys.executable, f"src/{step}"], check=True)
        print(f"-- {step} finished in {time.time() - t:.0f} s", flush=True)
    with open(MARKER, "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\nPipeline finished in {(time.time() - t0) / 60:.1f} minutes. Results are in results/")


if __name__ == "__main__":
    main()
