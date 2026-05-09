import time
import torch

def _sync_torch_if_needed(*xs):
    """
    Synchronize CUDA tensors before reading wall-clock timings.

    CPU tensors are ignored. This keeps timing measurements honest for CUDA
    kernels, which otherwise launch asynchronously.
    """
    for x in xs:
        if x.is_cuda:
            torch.cuda.synchronize(x.device)

def _qr_q(X):
    """
    Return a row-orthonormal basis for the row space of X.

    The QR factorization is applied to X.T and then transposed back so callers
    can keep using row-space iterates.
    """
    return torch.linalg.qr(X.mT, mode="reduced").Q.mT

def _run_svd(X):
    """
    Compute the reduced singular value decomposition of X.
    """
    return torch.linalg.svd(X, full_matrices=False)

def _as_like_tensor(X, like):
    """
    Move tensor X to the dtype and device of like when needed.

    Non-tensor inputs are returned unchanged, allowing callers to pass values
    that are already compatible with torch matrix operations.
    """
    if torch.is_tensor(X) and (X.dtype != like.dtype or X.device != like.device):
        return X.to(dtype=like.dtype, device=like.device)
    return X

def _compute_spectral_norm_error(A, Q, max_svd_iterations=50, tol=1e-6):
    """
    Estimate ||A(I - Q.T Q)||_2 by power iteration on the residual operator.

    Q is expected to have orthonormal rows. The returned value is a Python
    float, and gradient tracking is disabled because this is a diagnostic
    error calculation.
    """
    def residual_matvec(v):
        """
        Apply the right-projection residual A(I - Q.T Q) to v.
        """
        return A @ (v - Q.mT @ (Q @ v))

    def residual_rmatvec(u):
        """
        Apply the transpose of the right-projection residual to u.
        """
        z = A.mT @ u
        return z - Q.mT @ (Q @ z)

    with torch.no_grad():
        x = torch.randn(A.shape[1], dtype=A.dtype, device=A.device)
        x_norm = torch.linalg.vector_norm(x)
        if x_norm == 0:
            return 0.0
        x = x / x_norm

        prev_sigma = None
        sigma = torch.zeros((), dtype=A.real.dtype, device=A.device)
        for _ in range(max_svd_iterations):
            y = residual_matvec(x)
            sigma = torch.linalg.vector_norm(y)
            if sigma == 0:
                return 0.0

            z = residual_rmatvec(y)
            z_norm = torch.linalg.vector_norm(z)
            if z_norm == 0:
                return sigma.item()

            if prev_sigma is not None:
                change = torch.abs(sigma - prev_sigma)
                scale = torch.maximum(
                    torch.ones((), dtype=sigma.dtype, device=sigma.device),
                    torch.abs(prev_sigma),
                )
                if change <= tol * scale:
                    break

            x = z / z_norm
            prev_sigma = sigma

        sigma = torch.linalg.vector_norm(residual_matvec(x))
        return sigma.item()


def _compute_factored_spectral_norm_error(
    A, left_factor, right_factor, max_svd_iterations=50, tol=1e-6
):
    """
    Estimate ||A - left_factor @ right_factor||_2 by power iteration.

    The factors are never multiplied into a dense residual matrix. Instead,
    this routine applies the residual and its transpose as linear operators and
    returns the estimated spectral norm as a Python float.
    """
    def residual_matvec(v):
        """
        Apply the factored residual A - left_factor @ right_factor to v.
        """
        return A @ v - left_factor @ (right_factor @ v)

    def residual_rmatvec(u):
        """
        Apply the transpose of the factored residual to u.
        """
        return A.mT @ u - right_factor.mT @ (left_factor.mT @ u)

    with torch.no_grad():
        x = torch.randn(A.shape[1], dtype=A.dtype, device=A.device)
        x_norm = torch.linalg.vector_norm(x)
        if x_norm == 0:
            return 0.0
        x = x / x_norm

        prev_sigma = None
        sigma = torch.zeros((), dtype=A.real.dtype, device=A.device)
        for _ in range(max_svd_iterations):
            y = residual_matvec(x)
            sigma = torch.linalg.vector_norm(y)
            if sigma == 0:
                return 0.0

            z = residual_rmatvec(y)
            z_norm = torch.linalg.vector_norm(z)
            if z_norm == 0:
                return sigma.item()

            if prev_sigma is not None:
                change = torch.abs(sigma - prev_sigma)
                scale = torch.maximum(
                    torch.ones((), dtype=sigma.dtype, device=sigma.device),
                    torch.abs(prev_sigma),
                )
                if change <= tol * scale:
                    break

            x = z / z_norm
            prev_sigma = sigma

        sigma = torch.linalg.vector_norm(residual_matvec(x))
        return sigma.item()


def _iteration_errorswithiterate(
    A_error, A_iter, Q0, t, max_iterations=50, initial_elapsed=0.0, svd_param=False,
):
    """
    Shared implementation for timed QR-stabilized power iteration.

    The reported elapsed time for each iterate includes forming Q.T @ A_error.

    The factor-formation time is included only in the elapsed time recorded for
    the current iterate. It is not accumulated into the running total used to
    time the next iterate, matching low_rank_factorization_iteration_errorswithtime.
    """
    elapsed = initial_elapsed
    errors_with_time = []
    factor_costs = []

    step_start = time.perf_counter()
    Q = Q0
    Q = _qr_q(Q)
    _sync_torch_if_needed(Q)
    elapsed += time.perf_counter() - step_start

    factor_start = time.perf_counter()
    R =  A_error @ Q.mT
    _sync_torch_if_needed(R)
    if svd_param:
        U, S, Vt = _run_svd(R)
        _sync_torch_if_needed(U, S, Vt)
    factor_cost = time.perf_counter() - factor_start
    iterate_elapsed = elapsed + factor_cost
    factor_costs.append(factor_cost)
    iter_count = 0

    if iterate_elapsed >= t:
        print('Time Budget Insufficient')
        return errors_with_time

    error = _compute_spectral_norm_error(A_error, Q)
    errors_with_time.append((iter_count, elapsed, error))

    iter_costs = []
    while iter_count < max_iterations:
        step_start = time.perf_counter()
        Q =  Q @ A_iter.mT

        Q =  Q @ A_iter
        Q = _qr_q(Q)
        _sync_torch_if_needed(Q)
        iter_cost = time.perf_counter() - step_start
        elapsed += iter_cost
        iter_costs.append(iter_cost)

        factor_start = time.perf_counter()
        R =  A_error @ Q.mT
        _sync_torch_if_needed(R)
        if svd_param:
            U, S, Vt = _run_svd(R)
            _sync_torch_if_needed(U, S, Vt)
        factor_cost = time.perf_counter() - factor_start
        iterate_elapsed = elapsed + factor_cost
        factor_costs.append(factor_cost)

        iter_count += 1

        if iterate_elapsed >= t:
            break

        error = _compute_spectral_norm_error(A_error, Q)
        errors_with_time.append((iter_count, elapsed, error))

    avg_iter_cost = sum(iter_costs) / len(iter_costs) if iter_costs else float("nan")
    print(f"Average iteration cost = {avg_iter_cost} s")
    avg_factor_cost = sum(factor_costs) / len(factor_costs) if factor_costs else 0
    print(f"Average factor formation cost = {avg_factor_cost} s")

    errors_with_time_withfactor = [(a[0],a[1]+avg_factor_cost,a[2]) for a in errors_with_time]

    print(errors_with_time_withfactor)
    return errors_with_time_withfactor

def power_iteration_errorswithtime(
    A, S, t, max_iterations=10, svd_param=False
):
    """
    Run QR-stabilized power iteration on AA^T and track spectral error.


    Returns a list of (iteration_count, elapsed_time, error) tuples, starting with the
    zero-iteration approximation from qr(A @ S), and then one for each
    completed AA^T iteration whose multiply/QR time stays within the
    budget t. Factor-formation time for Q.T @ A is included in the
    reported elapsed time; spectral-error evaluation time is excluded.

    """
    t = float(t)
    S = _as_like_tensor(S, A)

    M = S @ A #to load relevant packages
    _sync_torch_if_needed(M)
    step_start = time.perf_counter()
    Q0 = S @ A
    _sync_torch_if_needed(Q0)
    setup_elapsed = time.perf_counter() - step_start
    print(f'Time taken for sketch = {setup_elapsed} secs')

    return _iteration_errorswithiterate(
        A_error=A,
        A_iter=A,
        Q0=Q0,
        t=t,
        max_iterations=max_iterations,
        initial_elapsed=setup_elapsed,
        svd_param=svd_param,
    )


def sketched_power_iteration_errorswithtime(
    A, S_1, S_2, t, max_iterations=10, svd_param=False
):
    """
    Run QR-stabilized row-space power iteration on the sketched matrix
    A_s^T A_s, where A_s = S_1 @ A, and track spectral error against A.

    The row-space basis is stored as a row-orthonormal tensor Q. 
    S_1 first sketches the rows of A, S_2 forms the initial row-space sketch
    from A_s, and subsequent iterations use A_s instead of A.
    """
    t = float(t)
    S_1 = _as_like_tensor(S_1, A)
    S_2 = _as_like_tensor(S_2, A)

    M1 = S_1 @ A #to load relevant packages
    _sync_torch_if_needed(M1)
    M2 = S_2 @ M1 #to load relevant packages
    _sync_torch_if_needed(M2)
    step_start = time.perf_counter()
    A_s = S_1 @ A
    _sync_torch_if_needed(A_s)
    mult1_end = time.perf_counter()

    Q0 = S_2 @ A_s
    _sync_torch_if_needed(Q0)
    mult2_end = time.perf_counter()

    setup_elapsed = mult2_end - step_start
    print(f'Big sketch mult time = {mult1_end-step_start}')
    print(f'Small sketch mult time = {mult2_end-mult1_end}')

    return _iteration_errorswithiterate(
        A_error=A,
        A_iter=A_s,
        Q0=Q0,
        t=t,
        max_iterations=max_iterations,
        initial_elapsed=setup_elapsed,
        svd_param=svd_param,
    )


def low_rank_factorization_iteration_errorswithtime(
    A,
    S_1,
    S_2,
    S_3,
    t,
    max_iterations=50,
    tol=1e-6,
):
    """
    Track row-sketched low-rank factorization spectral error with torch tensors.

    The sketches are applied as

        A_s = S_1 @ A,      S_1 in R^{s x m}
        X   = S_2 @ A_s,    S_2 in R^{k x s}
        R   = A @ S_3.T,    S_3 in R^{s x d}

    The approximation at each iterate is L @ X, where
    L = R @ pinv(X @ S_3.T). The returned tuples follow this module's
    convention: (iteration_count, elapsed_time, error).
    """
    t = float(t)
    S_1 = _as_like_tensor(S_1, A)
    S_2 = _as_like_tensor(S_2, A)
    S_3 = _as_like_tensor(S_3, A)
    A_t = A.mT.contiguous()

    M1 = S_1 @ A #to load relevant packages
    _sync_torch_if_needed(M1)
    M2 = S_2 @ M1 #to load relevant packages
    _sync_torch_if_needed(M2)
    step_start = time.perf_counter()
    A_s = S_1 @ A
    _sync_torch_if_needed(A_s)
    A_s_end = time.perf_counter()

    X = S_2 @ A_s
    _sync_torch_if_needed(X)
    X_end = time.perf_counter()

    X = _qr_q(X)
    _sync_torch_if_needed(X)

    R_start = time.perf_counter()
    #R = A @ S_3.mT
    R = (S_3 @ A_t).mT
    _sync_torch_if_needed(R)
    R_end = time.perf_counter()

    elapsed = R_end - step_start
    print(f"Time taken to form A_s = {A_s_end - step_start} secs")
    print(f"Time taken to form first X = {X_end - A_s_end} secs")
    print(f"Time taken to form R = {R_end - R_start} secs")

    factor_costs = []
    X_t = X.mT.contiguous()
    factor_start = time.perf_counter()
    #Y = X @ S_3.mT
    Y = (S_3 @ X_t).mT
    _sync_torch_if_needed(Y)
    Y_end = time.perf_counter()
    L = R @ torch.linalg.pinv(Y)
    _sync_torch_if_needed(L)
    factor_cost = time.perf_counter() - factor_start
    iterate_elapsed = elapsed + factor_cost
    print(f"Time taken to form first Y = {Y_end - factor_start} secs")
    factor_costs.append(factor_cost)

    iter_count = 0
    if iterate_elapsed >= t:
        print("Time Budget Insufficient")
        return []

    iteration_errors = [
        (
            iter_count,
            elapsed,
            _compute_factored_spectral_norm_error(
                A,
                L,
                X,
                tol=tol,
            ),
        )
    ]

    iter_costs = []
    while iter_count < max_iterations:
        step_start = time.perf_counter()
        X = X @ A_s.mT
        X = X @ A_s
        X = _qr_q(X)
        _sync_torch_if_needed(X)
        iter_cost = time.perf_counter() - step_start
        elapsed += iter_cost
        iter_costs.append(iter_cost)

        X_t = X.mT.contiguous()
        factor_start = time.perf_counter()
        #Y = X @ S_3.mT
        Y = (S_3 @ X_t).mT
        L = R @ torch.linalg.pinv(Y)
        _sync_torch_if_needed(Y, L)
        factor_cost = time.perf_counter() - factor_start
        iterate_elapsed = elapsed + factor_cost
        factor_costs.append(factor_cost)
        iter_count += 1

        if iterate_elapsed >= t:
            break

        iteration_errors.append(
            (
                iter_count,
                elapsed,
                _compute_factored_spectral_norm_error(
                    A,
                    L,
                    X,
                    tol=tol,
                ),
            )
        )

    avg_iter_cost = sum(iter_costs) / len(iter_costs) if iter_costs else float("nan")
    print(f"Average iteration cost = {avg_iter_cost} s")
    avg_factor_cost = sum(factor_costs) / len(factor_costs) if factor_costs else 0
    print(f"Average factor formation cost = {avg_factor_cost} s")

    iteration_errors_withfactor = [
        (iter_count, elapsed + avg_factor_cost, error)
        for iter_count, elapsed, error in iteration_errors
    ]

    print(iteration_errors_withfactor)
    return iteration_errors_withfactor

def nosketch_low_rank_factorization_iteration_errorswithtime(
    A,
    S_2,
    S_3,
    t,
    max_iterations=50,
    tol=1e-6,
):
    """
    Track row-sketched low-rank factorization spectral error with torch tensors.

     The sketches are applied as

        X   = S_2 @ A,    S_2 in R^{k x s}
        R   = A @ S_3.T,    S_3 in R^{s x d}

    The approximation at each iterate is L @ X, where
    L = R @ pinv(X @ S_3.T). The returned tuples follow this module's
    convention: (iteration_count, elapsed_time, error).
    """
    t = float(t)
    S_2 = _as_like_tensor(S_2, A)
    S_3 = _as_like_tensor(S_3, A)
    A_t = A.mT.contiguous()

    M2 = S_2 @ A #to load relevant packages
    _sync_torch_if_needed(M2)
    M3 = (S_3 @ A_t).mT #to load relevant packages
    _sync_torch_if_needed(M3)
    step_start = time.perf_counter()
    X = S_2 @ A
    _sync_torch_if_needed(X)
    X_end = time.perf_counter()

    X = _qr_q(X)
    _sync_torch_if_needed(X)

    R_start = time.perf_counter()
    #R = A @ S_3.mT
    R = (S_3 @ A_t).mT
    _sync_torch_if_needed(R)
    R_end = time.perf_counter()

    elapsed = R_end - step_start
    print(f"Time taken to form first X = {X_end - step_start} secs")
    print(f"Time taken to form R = {R_end - R_start} secs")

    factor_costs = []
    X_t = X.mT.contiguous()
    factor_start = time.perf_counter()
    #Y = X @ S_3.mT
    Y = (S_3 @ X_t).mT
    _sync_torch_if_needed(Y)
    Y_end = time.perf_counter()
    L = R @ torch.linalg.pinv(Y)
    _sync_torch_if_needed(L)
    factor_cost = time.perf_counter() - factor_start
    iterate_elapsed = elapsed + factor_cost
    print(f"Time taken to form first Y = {Y_end - factor_start} secs")
    factor_costs.append(factor_cost)

    iter_count = 0
    if iterate_elapsed >= t:
        print("Time Budget Insufficient")
        return []

    iteration_errors = [
        (
            iter_count,
            elapsed,
            _compute_factored_spectral_norm_error(
                A,
                L,
                X,
                tol=tol,
            ),
        )
    ]

    iter_costs = []
    while iter_count < max_iterations:
        step_start = time.perf_counter()
        X = X @ A.mT
        X = X @ A
        X = _qr_q(X)
        _sync_torch_if_needed(X)
        iter_cost = time.perf_counter() - step_start
        elapsed += iter_cost
        iter_costs.append(iter_cost)

        X_t = X.mT.contiguous()
        factor_start = time.perf_counter()
        #Y = X @ S_3.mT
        Y = (S_3 @ X_t).mT
        L = R @ torch.linalg.pinv(Y)
        _sync_torch_if_needed(Y, L)
        factor_cost = time.perf_counter() - factor_start
        iterate_elapsed = elapsed + factor_cost
        factor_costs.append(factor_cost)
        iter_count += 1

        if iterate_elapsed >= t:
            break

        iteration_errors.append(
            (
                iter_count,
                elapsed,
                _compute_factored_spectral_norm_error(
                    A,
                    L,
                    X,
                    tol=tol,
                ),
            )
        )

    avg_iter_cost = sum(iter_costs) / len(iter_costs) if iter_costs else float("nan")
    print(f"Average iteration cost = {avg_iter_cost} s")
    avg_factor_cost = sum(factor_costs) / len(factor_costs) if factor_costs else 0
    print(f"Average factor formation cost = {avg_factor_cost} s")

    iteration_errors_withfactor = [
        (iter_count, elapsed + avg_factor_cost, error)
        for iter_count, elapsed, error in iteration_errors
    ]

    print(iteration_errors_withfactor)
    return iteration_errors_withfactor


def _as_float(x):
    """
    Convert a scalar tensor or numeric value to a Python float.
    """
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)

def _mean_or_nan(values):
    """
    Return the arithmetic mean of values, or NaN for an empty sequence.
    """
    return sum(values) / len(values) if values else float("nan")

def _normalize_iteration_curve(iteration_errors, denom):
    """
    Normalize raw iteration errors by a reference denominator.

    Each output tuple is (iteration_count, elapsed_time, error / denom - 1).
    """
    denom = _as_float(denom)
    return [
        (int(iter_count), float(elapsed), (_as_float(error) / denom) - 1)
        for iter_count, elapsed, error in iteration_errors
    ]

def _average_iteration_curves(trial_curves):
    """
    Average elapsed times and normalized errors across trial curves.

    Curves are grouped by iteration count. Missing values for an iteration are
    ignored, and empty groups produce NaN through _mean_or_nan.
    """
    max_iter_count = -1
    grouped_elapsed = {}
    grouped_errors = {}

    for trial_curve in trial_curves:
        for iter_count, elapsed, error in trial_curve:
            max_iter_count = max(max_iter_count, iter_count)
            grouped_elapsed.setdefault(iter_count, []).append(elapsed)
            grouped_errors.setdefault(iter_count, []).append(error)

    time_points = []
    average_errors = []
    avg_error_messages = []
    for iter_count in range(max_iter_count + 1):
        avg_elapsed = _mean_or_nan(grouped_elapsed.get(iter_count, []))
        avg_error = _mean_or_nan(grouped_errors.get(iter_count, []))
        time_points.append(float(avg_elapsed))
        average_errors.append(float(avg_error))
        avg_error_messages.append(
            f"iteration {iter_count}: time {avg_elapsed}, error {avg_error}"
        )

    if avg_error_messages:
        print("Average per-iteration errors | " + " | ".join(avg_error_messages))
    return time_points, average_errors


def run_power_iteration_trials(
    sketch_fn,
    A,
    D,
    k,
    t,
    num_trials,
    average=True,
    max_iterations=5,
    svd_param=False,
    **sketch_kwargs,
):
    """
    Run row-space power-iteration trials and return per-iteration errors.

    Sketches are applied on the left, so sketch_fn is called as
    sketch_fn(k, A.shape[0], **sketch_kwargs).
    """

    m = A.shape[0]
    denom = D[k]
    trial_curves = []
    for _ in range(num_trials):
        iteration_errors = power_iteration_errorswithtime(
            A,
            sketch_fn(k, m, format='t', **sketch_kwargs),
            t,
            max_iterations=max_iterations,
            svd_param=svd_param,
        )
        trial_curves.append(_normalize_iteration_curve(iteration_errors, denom))

    if average:
        return _average_iteration_curves(trial_curves)
    return trial_curves


def run_sketched_power_iteration_trials(
    sketch_fn_1,
    sketch_fn_2,
    A,
    D,
    k,
    s,
    t,
    num_trials,
    sketch_kwargs_1=None,
    sketch_kwargs_2=None,
    average=False,
    max_iterations=50,
    svd_param=False,
):
    """
    Run row-space sketched power-iteration trials and return per-iteration errors.

    The left sketches are generated as S_1 in R^{s x m} and S_2 in R^{k x s}.
    """


    
    m = A.shape[0]
    denom = D[k]
    sketch_kwargs_1 = {} if sketch_kwargs_1 is None else dict(sketch_kwargs_1)
    sketch_kwargs_2 = {} if sketch_kwargs_2 is None else dict(sketch_kwargs_2)

    trial_curves = []
    for _ in range(num_trials):
        iteration_errors = sketched_power_iteration_errorswithtime(
            A,
            sketch_fn_1(s, m, **sketch_kwargs_1),
            sketch_fn_2(k, s, format='t', **sketch_kwargs_2),
            t,
            max_iterations=max_iterations,
            svd_param=svd_param,
        )
        trial_curves.append(_normalize_iteration_curve(iteration_errors, denom))

    if average:
        return _average_iteration_curves(trial_curves)
    return trial_curves


def run_low_rank_factorization_trials(
    sketch_fn_1,
    sketch_fn_2,
    sketch_fn_3,
    A,
    D,
    k,
    s,
    t,
    num_trials,
    sketch_kwargs_1=None,
    sketch_kwargs_3=None,
    average=False,
    max_iterations=5,
    tol=1e-6,
):
    """
    Run row-sketched low-rank factorization trials with torch sketches.

    The sketches are generated as S_1 in R^{s x m}, S_2 in R^{k x s},
    and S_3 in R^{s x d}. The output matches the other row-torch trial
    helpers: normalized per-iteration curves, or their average.
    """

    m, d = A.shape
    denom = D[k]

    trial_curves = []
    for _ in range(num_trials):
        iteration_errors = low_rank_factorization_iteration_errorswithtime(
            A,
            sketch_fn_1(s, m, **sketch_kwargs_1),
            sketch_fn_2(k, s, format='t'),
            sketch_fn_3(s, d, **sketch_kwargs_3),
            t,
            max_iterations=max_iterations,
            tol=tol,
        )
        trial_curves.append(_normalize_iteration_curve(iteration_errors, denom))

    if average:
        return _average_iteration_curves(trial_curves)
    return trial_curves

def run_nosketch_low_rank_factorization_trials(
    sketch_fn_2,
    sketch_fn_3,
    A,
    D,
    k,
    s,
    t,
    num_trials,
    sketch_kwargs_3=None,
    average=False,
    max_iterations=5,
    tol=1e-6,
):
    """
    Run row-sketched low-rank factorization trials with torch sketches.

    The sketches are generated as S_2 in R^{k x s},
    and S_3 in R^{s x d}. The output matches the other row-torch trial
    helpers: normalized per-iteration curves, or their average.
    """

    m, d = A.shape
    denom = D[k]

    trial_curves = []
    for _ in range(num_trials):
        iteration_errors = nosketch_low_rank_factorization_iteration_errorswithtime(
            A,
            sketch_fn_2(k, m, format='t'),
            sketch_fn_3(s, d, **sketch_kwargs_3),
            t,
            max_iterations=max_iterations,
            tol=tol,
        )
        trial_curves.append(_normalize_iteration_curve(iteration_errors, denom))

    if average:
        return _average_iteration_curves(trial_curves)
    return trial_curves
