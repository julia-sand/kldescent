# Testing

This package has a short test suite to prevent regressions with new code changes. 

## Running Tests

The test runs the training for the UnderdampedGaussian boundary conditions (`tests/train_from_yaml.py`) using the configuration in `gaussian_test_params.yaml` and compares the final position variance against the known benchmark value (`var_bench`).

### Training from YAML Test

To run this test, execute the following command from the root of the repository:

```bash
pytest tests
```
