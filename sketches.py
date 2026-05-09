import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as spar
from scipy.fft import dct, idct
import scipy.sparse.linalg as sparla
import math
try:
    import torch
except ModuleNotFoundError:
    torch = None


def _torch_sparse_coo_from_scipy(S):
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for format='t' sparse sketches."
        )
    S = S.tocoo()
    indices = torch.from_numpy(
        np.vstack((S.row, S.col)).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(np.asarray(S.data))
    return torch.sparse_coo_tensor(indices, values, S.shape).coalesce()


def sparse_signs_iid(n_rows, n_cols, rng=None, density=0.05, format='r'):
    # get row indices and col indices
    rng = np.random.default_rng(rng)
    nonzero_idxs = rng.random(n_rows * n_cols) < density
    attempt = 0
    while np.all(~nonzero_idxs):
        if attempt == 10:
            raise RuntimeError('Density too low.')
        nonzero_idxs = rng.random(n_rows * n_cols) < density
        attempt += 1
    nonzero_idxs = np.where(nonzero_idxs)[0]
    rows, cols = np.unravel_index(nonzero_idxs, (n_rows, n_cols))
    # get values for each row and col index
    nnz = rows.size
    vals = np.ones(nnz)
    vals[rng.random(vals.size) < 0.5] = -1
    vals /= np.sqrt(min(n_rows, n_cols) * density)
    # Wrap up
    S = spar.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
    if format == 'c':
        S = S.tocsc()
    elif format == 'r':
        S = S.tocsr()
    elif format == 't':
        S = _torch_sparse_coo_from_scipy(S)
    elif format == 'o':
        pass
    else:
        raise ValueError("format must be either 'r', 'c', 't' or 'o'.")
    return S

def sparse_signs_fixednnz(n_row, n_cols, nnz_per=1, rng=None, nnz_axis='r', format='r'):
    """
    Generate a sparse sign embedding with exactly k nonzeros along one axis.

    Parameters
    ----------
    nnz_axis : {'r', 'c'}
        Use 'r' for exactly k nonzeros per row, or 'c' for exactly k
        nonzeros per column.
    format : {'r', 'c'}
        Use 'r' to return CSR format, or 'c' to return CSC format.
    """
    k = nnz_per
    k = math.ceil(k)
    if k <= 0:
        raise ValueError('k must be positive.')

    rng = np.random.default_rng(rng)

    if nnz_axis == 'c':
        if k > n_row:
            raise ValueError(
                'k cannot exceed n_row when sampling without replacement.'
            )

        cols = np.repeat(np.arange(n_cols), k)
        if k == 1:
            rows = rng.integers(0, n_row, size=n_cols)
        else:
            rows = np.empty(n_cols * k, dtype=np.int64)
            for col in range(n_cols):
                start = col * k
                rows[start:start + k] = rng.choice(n_row, size=k, replace=False)
    elif nnz_axis == 'r':
        if k > n_cols:
            raise ValueError(
                'k cannot exceed n_cols when sampling without replacement.'
            )

        rows = np.repeat(np.arange(n_row), k)
        if k == 1:
            cols = rng.integers(0, n_cols, size=n_row)
        else:
            cols = np.empty(n_row * k, dtype=np.int64)
            for row in range(n_row):
                start = row * k
                cols[start:start + k] = rng.choice(n_cols, size=k, replace=False)
    else:
        raise ValueError("nnz_axis must be either 'r' or 'c'.")

    vals = rng.choice(np.array([-1.0, 1.0]), size=rows.size) / np.sqrt(k)
    S = spar.coo_matrix((vals, (rows, cols)), shape=(n_row, n_cols))

    if format == 'c':
        S = S.tocsc()
    elif format == 'r':
        S = S.tocsr()
    elif format == 't':
        S = _torch_sparse_coo_from_scipy(S)
    elif format == 'o':
        pass
    else:
        raise ValueError("format must be either 'r', 'c', 't' or 'o'.")
        
    return S


def gaussian(n_rows, n_cols, rng=None, format='n', dtype=None, device=None):
    rng = np.random.default_rng(rng)
    S = rng.standard_normal((n_rows, n_cols)) / np.sqrt(min(n_rows, n_cols))
    if format == 'n':
        return S
    if format == 't':
        if torch is None:
            raise ModuleNotFoundError(
                "torch is required for format='t' Gaussian sketches."
            )
        return torch.as_tensor(S, dtype=dtype, device=device)
    raise ValueError("format must be either 'n' or 't'.")
