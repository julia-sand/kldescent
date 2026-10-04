# Testing

This package includes a test suite to verify the correctness of the training process.

## Running Tests

The main test for the training utility verifies that `train_from_yaml.py` produces results consistent with known benchmarks.

### Training from YAML Test

This test runs the training process using the configuration in `gaussian_test_params.yaml` and compares the final position variance against the benchmark value (`var_bench`) specified in the same YAML file.

To run this test, execute the following command from the root of the repository:

```bash
python3 tests/test_train_from_yaml.py
```

### What the test does:
1. Loads the benchmark variance from `gaussian_test_params.yaml`.
2. Executes `train_from_yaml.py` with the specified parameters.
3. Parses the training output to extract the final position variance.
4. Calculates the relative error between the trained result and the benchmark.
5. Passes if the relative error is within a 10% tolerance.

## Adding New Tests
New tests should be placed in the `tests/` directory. Ensure that any paths to source files or configuration files are handled relative to the project root to maintain portability.
