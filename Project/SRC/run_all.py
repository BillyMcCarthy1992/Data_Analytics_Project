# run_all.py

from pathlib import Path
import subprocess
import sys
import time

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
base_dir = project_dir.parent

scripts = [
    "CSV_retrieval.py",
    "inspect_file.py",
    "data_quality.py",
    "eda.py",
    "preprocessing.py",
    "train_baseline_models.py",
    "further_models.py",
    "fine_tuning_F1.py",
    "fine_tuning_recall.py",
    "report_graphics.py",
    "train_models_weighted.py",
    "explainability.py",
    "SHAP.py",
    "interpretability_comparison.py",
    "barchart.py",
    "comparison_heatmap.py",
    "directional_heatmap.py"
    
]

print("Starting full project run...")
print(f"SRC folder: {script_dir}")
print(f"Project folder: {project_dir}")
print(f"Working directory: {base_dir}")
print()

for i, script_name in enumerate(scripts, start=1):
    script_path = script_dir / script_name

    if not script_path.exists():
        print(f"Could not find script: {script_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"Step {i}/{len(scripts)}: running {script_name}")
    print("=" * 60)

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=base_dir
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print()
        print(f"{script_name} failed.")
        print(f"Stopped at step {i}.")
        print(f"Return code: {result.returncode}")
        sys.exit(result.returncode)

    print()
    print(f"Finished {script_name} in {elapsed:.2f} seconds.")
    print()

print("=" * 60)
print("Whole project finished successfully.")
print("=" * 60)