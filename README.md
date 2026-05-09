# Sketched Power Method

Experiments for the paper 'Accelerating Power Method with Fast Sketching
for Stronger Low-Rank Approximation'

The experiments use PyTorch tensors for the main linear algebra and support CUDA when available.

## Repository Layout

- `algorithms.py` contains the core trial routines, timing logic, QR-stabilized power iteration, and spectral-error estimators.
- `sketches.py` contains random sketch constructors: Gaussian sketches, iid sparse sign sketches, and fixed-nonzero sparse sign embeddings.
- `helpers.py` contains plotting utilities for grouped error curves.
- `processing.ipynb` prepares experiment matrices and singular values.
- `run_trials.ipynb` runs the benchmark trials and saves plots.


## Data Preparation

Use `processing.ipynb` to build the matrices consumed by the trials. At the top of the notebook, set:

```python
DATA_MAIN_FOLDER = "/path/to/raw/data/"
DATA_SAVE_FOLDER = "/path/to/output/matrices/"
```

The notebook can produce:

- `smallnorb.npz` from the small NORB training data file.
- `imagenet.npz` from an image directory containing ImageNet images, using resized red-channel image vectors.
- `polydecay.npz`, a synthetic matrix with polynomially decaying singular values.

Each saved `.npz` file contains:

- `A`: the matrix used in the experiment.
- `D`: singular values in decreasing order, used to normalize approximation errors.

## Running Experiments

Open `run_trials.ipynb` and set:

```python
MATRICES_DIRECTORY = "/path/to/output/matrices/"
OUTPUT_DIRECTORY = "Plots"
MATRIX_NAMES = ["smallnorb", "polydecay", "imagenet"]
```

Then run either of the notebook experiment drivers.

For low-rank factorization comparisons:

```python
lrfactors(
    k=40,
    t=10,
    num_trials=25,
    int_sizes=[800, 1200, 1600],
    max_iterations=5,
    sparse_signs_nnz=1,
    save_png=True,
    output_dir=OUTPUT_DIRECTORY,
)
```

For randomized SVD comparisons:

```python
randsvd(
    k=40,
    t=10,
    num_trials=25,
    int_sizes=[800, 1200, 1600],
    max_iterations=5,
    sparse_signs_nnz=1,
    save_png=True,
    output_dir=OUTPUT_DIRECTORY,
)
```

Plots are written as PNG files under `OUTPUT_DIRECTORY` when `save_png=True`.

## Notes

- Timings exclude spectral-error evaluation, which is treated as diagnostics.
- CUDA kernels are synchronized before wall-clock timings are recorded.
- The spectral norm errors are estimated by power iteration rather than by explicitly forming large residual matrices.
- The outputs of the notebook cells in 'run_trials.ipynb' contain the plots used in the paper; you may rerun the notebooks after changing paths, devices, or trial parameters.
