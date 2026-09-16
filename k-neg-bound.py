import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from math import comb
import pickle
import os
import time

def partial_transpose_B(rho, d):
    """Partial transpose on subsystem B for a density matrix rho (d^2 x d^2)."""
    rho_4d = rho.reshape(d, d, d, d)
    rho_pt = rho_4d.transpose(0, 3, 2, 1)
    return rho_pt.reshape(d*d, d*d)

def witness_from_projector(P, d):
    """Compute the witness operator W = -P^Gamma."""
    return -partial_transpose_B(P, d)

def projector_from_state(rho, k, d):
    """
        Construct a rank-k projector from the strictly negative eigenvectors of rho^Gamma.
        If fewer than k negative eigenvalues exist, the projector rank is reduced accordingly.
        """
    rho_pt = partial_transpose_B(rho, d)
    eigvals, eigvecs = eigh(rho_pt)
    neg_mask = eigvals < -1e-10  # numerical tolerance
    neg_count = np.sum(neg_mask)

    actual_k = min(k, neg_count)
    neg_idx = np.argsort(eigvals)[:actual_k]
    vecs = eigvecs[:, neg_idx]
    P = vecs @ vecs.conj().T
    return P

# ==================== Seesaw  ====================
def seesaw_single(p, k, d, n_init=5, max_iter=100, tol=1e-6):
    """
        Run Seesaw for fixed p, k, d with multiple random initializations.
        Returns the best objective value and the iteration traces for all initializations.
        """
    D = d * d
    best_val = -np.inf
    all_init_traces = []

    for init_idx in range(n_init):
        # Random rank-k projector initialization
        random_mat = np.random.randn(D, k) + 1j * np.random.randn(D, k)
        Q, _ = np.linalg.qr(random_mat)
        P = Q[:, :k] @ Q[:, :k].conj().T

        prev_val = -np.inf
        val = -np.inf
        iter_vals = []

        for it in range(max_iter):
            W = witness_from_projector(P, d)
            rho_var = cp.Variable((D, D), hermitian=True)
            constraints = [
                rho_var >> 0,
                cp.trace(rho_var) == 1.0,
                cp.norm(rho_var, 'fro') <= np.sqrt(p)
            ]
            objective = cp.Maximize(cp.real(cp.trace(rho_var @ W)))
            prob = cp.Problem(objective, constraints)

            try:
                prob.solve(solver='MOSEK', verbose=False)
                if prob.status in ['optimal', 'optimal_inaccurate']:
                    rho_new = rho_var.value
                    val = np.real(np.trace(rho_new @ W))
                else:
                    break
            except Exception:
                break

            iter_vals.append(val)

            P_new = projector_from_state(rho_new, k, d)
            if abs(val - prev_val) < tol:
                break
            P = P_new
            prev_val = val

        all_init_traces.append(iter_vals)
        if val > best_val:
            best_val = val

    return (best_val if best_val > -np.inf else 0.0), all_init_traces

# ==================== Adaptive dimension increase with retry ====================
def compute_Nk(p, k, d_start=None, n_init=5, tol=1e-5, max_retries=1):
    if d_start is None:
        d = k + 1
    else:
        d = d_start
    prev_val = -1.0
    final_val = 0.0
    final_d = d
    d_trials = []
    retry_count = 0

    while True:
        # 当前维度正常求解
        val, init_traces = seesaw_single(p, k, d, n_init=n_init)
        d_trials.append({"d": d, "val": val, "init_traces": init_traces})
        print(f"  p={p:.3f}, k={k}, d={d}, val={val:.6f}, n_init={n_init}")

        # First dimension (d == k+1) always accepted
        if d <= k + 1:
            prev_val = val
            d += 1
            continue

        # Check improvement after increasing d
        if val - prev_val < tol:
            if retry_count < max_retries:
                # Fall back to previous dimension and retry with more initializations
                d -= 1
                n_init_retry = n_init * 2
                val_retry, init_retry = seesaw_single(p, k, d, n_init=n_init_retry)
                d_trials.append({"d": d, "val": val_retry, "init_traces": init_retry, "retry": True})
                print(f"  retry d={d}, val={val_retry:.6f}, n_init={n_init_retry}")

                if val_retry - prev_val >= tol:
                    # Found better value after retry -> continue increasing
                    prev_val = val_retry
                    d += 1
                    n_init = n_init_retry
                    retry_count = 0
                else:
                    # Still no improvement -> converged
                    final_val = prev_val
                    final_d = d
                    break
            else:
                final_val = prev_val
                final_d = d - 1
                break
        else:
            prev_val = val
            d += 1
            retry_count = 0

        if d > 12:   # safety upper bound
            final_val = prev_val
            final_d = d - 1
            break

    return final_val, final_d, d_trials
# ==================== Pure-state analytic maximum ====================
def lambda_max_core_spike(m, r):
    """Spectral radius of the core-spike graph."""
    if r == 0:
        return m - 1
    coeff = [1, -(m-2), -(m+r-1), r*(m-r-1)]
    roots = np.roots(coeff)
    real_roots = roots[np.isreal(roots)].real
    return np.max(real_roots)

def pure_Nk_max(k):
    """Maximum Ky Fan k-negativity for pure states."""
    m = 2
    while comb(m, 2) <= k:
        m += 1
    m -= 1
    r = k - comb(m, 2)
    return lambda_max_core_spike(m, r) / 2.0
# ==================== Exact N1 bound for mixed states====================
def exact_N1(p):
    if p >= 1.0:
        return 0.5
    S = np.sqrt(1.0 + 8.0 / p)
    v = int(np.ceil((1.0 + S) / 2.0))
    if v == 2:
        return 0.5
    n1 = (v - 1) * (v - 2) / 2.0
    n2 = v - 1.0
    n = v * (v - 1) / 2.0
    inner = (n2 / n1) * (p * n - 1.0)
    if inner < 0:
        inner = 0.0
    sqrt_inner = np.sqrt(inner)
    m1 = (1.0 / n) * (1.0 + sqrt_inner)
    m2 = (1.0 / n) * (1.0 - (n1 / n2) * sqrt_inner)
    lambda_max = ((v - 2) * m1 + np.sqrt((v - 2)**2 * m1**2 + 4 * (v - 1) * m2**2)) / 2.0
    return lambda_max / 2.0

# ==================== Main program ====================
if __name__ == "__main__":

    p_values = np.linspace(0.1, 1.0, 70)  # purity from low to high
    ks = [1, 2, 3, 4, 5, 6]
    data_file = "seesaw_results.pkl"
    traces = {}
    # ---------- 1. Load or compute data ----------
    if os.path.exists(data_file):
        print(f"Loading existing results from '{data_file}' ...")
        with open(data_file, "rb") as f:
            results, dims_used = pickle.load(f)
        print("Loading completed.")
    else:
        results = {k: [0.0] * len(p_values) for k in ks}
        dims_used = {k: [0] * len(p_values) for k in ks}
        current_d = {k: None for k in ks}

        print("Starting computation (from high to low purity) ...")
        t_start = time.time()

        for i in range(len(p_values) - 1, -1, -1):
            p = p_values[i]
            traces[i] = {}
            print(f"\nPurity p = {p:.4f}  ({len(p_values) - i}/{len(p_values)})")
            for k in ks:
                val, d_final, d_trials = compute_Nk(p, k, d_start=current_d[k], n_init=8)
                results[k][i] = val
                dims_used[k][i] = d_final
                current_d[k] = d_final
                traces[i][k] = d_trials

        t_end = time.time()
        print(f"\nComputation finished, total time {t_end - t_start:.1f} s")

        # Save final results
        with open(data_file, "wb") as f:
            pickle.dump((results, dims_used), f)
        print(f"Results saved to '{data_file}'.")

        # Save traces
        with open("seesaw_traces.pkl", "wb") as f:
            pickle.dump(traces, f)
        print(f"Traces saved to 'seesaw_traces.pkl'.")

    # ---------- 2. Plotting ----------
    plt.rcParams.update({
        # "text.usetex": True,
        # "font.family": "serif",
        # "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 16,
        "font.size": 12,
        "legend.fontsize": 14,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })
    plt.figure(figsize=(8, 5))
    colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']




    for idx, k in enumerate(ks):
        plt.plot(p_values, results[k], color=colors[idx], linewidth=2.5, zorder=1,
                 label=f'$\\mathcal{{N}}_{k}^{{*}}(p)$')
    N1_exact = [exact_N1(p) for p in p_values]
    plt.plot(p_values, N1_exact, 'k--', linewidth=2.2, alpha=0.5,
             label=' $\\mathcal{N}_{1}^{\\max}(p)$')
    # Exact N1 bound for mixed states
    pure_vals = [pure_Nk_max(k) for k in ks]
    plt.scatter([1.0] * len(ks), pure_vals, color='black', marker='*', s=100,
                edgecolors='black', linewidth=0.5, zorder=5, label=' $\\mathcal{N}_{k}^{\\max}$')

    plt.xlabel('Purity $p$')
    plt.ylabel('$\\mathcal{N}_{k}^{*}(p)$')
    # plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='center left',
               bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('seesaw_Nk.pdf', dpi=300)
    plt.show()