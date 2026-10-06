#!/usr/bin/env python
"""
Test suite to verify that train_from_yaml.py produces the expected position variance for the UnderdampedGaussian boundaries
for a given set of parameters.

Ensures that changes do not regress network training behaviour.
"""

import subprocess
import sys
import os
import re
import yaml

def test_train_from_yaml():
    # Paths relative to the project root
    project_root = os.path.dirname(__file__)#os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(project_root, "train_from_yaml.py")
    params_path = os.path.join(project_root, "gaussian_test_params.yaml")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Error: {script_path} not found")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Error: {params_path} not found")

    # Load benchmark variance from YAML
    with open(params_path, 'r') as f:
        cfg = yaml.safe_load(f)
    var_bench = cfg.get('var_bench')
    if var_bench is None:
        raise KeyError("Error: var_bench not found in params YAML")

    print(f"Running train_from_yaml.py with {params_path}...")
    print(f"Expected benchmark variance: {var_bench:.5f}")

    # Run the script
    env = os.environ.copy()
    # Ensure src is in PYTHONPATH
    src_path = os.path.join(project_root, "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path, params_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=600 # 10 minutes timeout
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError("Training timed out")

    if result.returncode != 0:
        print("Error: train_from_yaml.py failed")
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        raise RuntimeError(f"train_from_yaml.py failed with return code {result.returncode}")

    # Parse position variance from output
    match = re.search(r"position variance at Tf = ([\d.eE+-]+)", result.stdout)
    if not match:
        print("Error: Could not find position variance in output")
        print("Stdout:", result.stdout)
        raise RuntimeError("Could not find position variance in output")

    var_nn = float(match.group(1))
    print(f"Trained position variance: {var_nn:.5f}")

    # Calculate relative error
    rel_error = abs(var_nn - var_bench) / var_bench
    print(f"Relative error: {rel_error:.4f}")

    # Define a reasonable tolerance
    tolerance = 0.1 
    if rel_error < tolerance:
        print("Test PASSED")
    else:
        print(f"Test FAILED: Relative error {rel_error:.4f} exceeds tolerance {tolerance}")
        raise AssertionError(f"Relative error {rel_error:.4f} exceeds tolerance {tolerance}")

if __name__ == "__main__":
    test_train_from_yaml()
