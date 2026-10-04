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
        print(f"Error: {script_path} not found")
        sys.exit(1)
    if not os.path.exists(params_path):
        print(f"Error: {params_path} not found")
        sys.exit(1)

    # Load benchmark variance from YAML
    with open(params_path, 'r') as f:
        cfg = yaml.safe_load(f)
    var_bench = cfg.get('var_bench')
    if var_bench is None:
        print("Error: var_bench not found in params YAML")
        sys.exit(1)

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
        print("Error: Training timed out")
        sys.exit(1)

    if result.returncode != 0:
        print("Error: train_from_yaml.py failed")
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        sys.exit(1)

    # Parse position variance from output
    match = re.search(r"position variance at Tf = ([\\d.eE+-]+)", result.stdout)
    if not match:
        print("Error: Could not find position variance in output")
        print("Stdout:", result.stdout)
        sys.exit(1)

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
        sys.exit(1)

if __name__ == "__main__":
    test_train_from_yaml()
