import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from config import FIGURES_DIR, N_RFF_DIMS

# ─── Publication-quality style ───
COLORS = {
    'SVM-RBF': '#2196F3',
    'SVM-Poly': '#FF9800',
    'SVM-Linear': '#4CAF50',
    'QK-Informed': '#E53935',
    'QK-Generic': '#7B1FA2',
    'QK-Random': '#795548',
}

def _pub_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
    })

def _color(name):
    return COLORS.get(name, '#333333')


def analyze_eigenspectrum(kernel_matrices, save_path=None):
    _pub_style()
    eigenvalues = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1 = axes[0]
    for name, K in kernel_matrices.items():
        K_sym = (K + K.T) / 2
        eigvals = np.linalg.eigvalsh(K_sym)[::-1]
        eigvals = eigvals / eigvals[0]
        eigenvalues[name] = eigvals
        ax1.semilogy(range(1, len(eigvals) + 1), np.maximum(eigvals, 1e-15),
                     label=name, linewidth=1.5, color=_color(name))

    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Normalized Eigenvalue (log scale)')
    ax1.set_title('Kernel Matrix Eigenspectrum')
    ax1.legend(fontsize=8, frameon=False)

    ax2 = axes[1]
    names, eff_ranks = [], []
    for name, eigvals in eigenvalues.items():
        eigvals_pos = eigvals[eigvals > 0]
        p = eigvals_pos / eigvals_pos.sum()
        eff_rank = np.exp(-np.sum(p * np.log(p + 1e-15)))
        names.append(name)
        eff_ranks.append(eff_rank)

    colors = [_color(n) for n in names]
    bars = ax2.bar(range(len(names)), eff_ranks, color=colors,
                   edgecolor='black', linewidth=0.5, alpha=0.85, width=0.6)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax2.set_ylabel('Effective Rank')
    ax2.set_title('Kernel Effective Dimensionality')
    for bar, val in zip(bars, eff_ranks):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Also save PDF
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    return eigenvalues


def compute_kta(K, y):
    y = np.array(y, dtype=float)
    if set(np.unique(y)).issubset({0, 1}):
        y = 2 * y - 1
    yy = np.outer(y, y)
    numerator = np.sum(K * yy)
    denominator = np.sqrt(np.sum(K * K) * np.sum(yy * yy))
    if denominator < 1e-15:
        return 0.0
    return numerator / denominator


def compute_centered_kta(K, y):
    n = K.shape[0]
    ones = np.ones((n, n)) / n
    K_c = K - ones @ K - K @ ones + ones @ K @ ones
    y = np.array(y, dtype=float)
    if set(np.unique(y)).issubset({0, 1}):
        y = 2 * y - 1
    yy = np.outer(y, y)
    yy_c = yy - ones @ yy - yy @ ones + ones @ yy @ ones
    num = np.sum(K_c * yy_c)
    denom = np.sqrt(np.sum(K_c ** 2) * np.sum(yy_c ** 2))
    if denom < 1e-15:
        return 0.0
    return num / denom


def plot_kta_comparison(kta_results, save_path=None):
    _pub_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    kernel_names = list(kta_results.keys())
    task_names = list(next(iter(kta_results.values())).keys())
    n_kernels = len(kernel_names)
    n_tasks = len(task_names)
    x = np.arange(n_kernels)
    width = 0.8 / n_tasks

    for i, task in enumerate(task_names):
        values = [kta_results[k][task] for k in kernel_names]
        offset = (i - n_tasks / 2 + 0.5) * width
        colors = [_color(k) for k in kernel_names]
        ax.bar(x + offset, values, width, label=task, color=colors,
               edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(kernel_names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Kernel-Target Alignment')
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def rff_approximate_kernel(K_quantum, X, n_rff_dims_list=None, seed=42):
    if n_rff_dims_list is None:
        n_rff_dims_list = N_RFF_DIMS

    rng = np.random.RandomState(seed)
    n, d = X.shape

    from sklearn.metrics.pairwise import rbf_kernel
    best_gamma = 1.0 / d
    best_match = np.inf
    for gamma in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 1.0/d, 0.5/d, 2.0/d]:
        K_rbf = rbf_kernel(X, gamma=gamma)
        err = np.linalg.norm(K_quantum - K_rbf, 'fro') / np.linalg.norm(K_quantum, 'fro')
        if err < best_match:
            best_match = err
            best_gamma = gamma

    results = {
        'best_gamma': best_gamma,
        'best_rbf_match': best_match,
        'rff_dims': [],
        'rff_errors': [],
        'rbf_error': best_match,
    }

    for D in n_rff_dims_list:
        W = rng.randn(d, D) * np.sqrt(2 * best_gamma)
        b = rng.uniform(0, 2 * np.pi, D)
        Z = np.sqrt(2.0 / D) * np.cos(X @ W + b)
        K_rff = Z @ Z.T
        err = np.linalg.norm(K_quantum - K_rff, 'fro') / np.linalg.norm(K_quantum, 'fro')
        results['rff_dims'].append(D)
        results['rff_errors'].append(err)

    return results


def plot_dequantization(rff_results_dict, save_path=None):
    _pub_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, res in rff_results_dict.items():
        ax.plot(res['rff_dims'], res['rff_errors'],
                'o-', label=f'{name} (best RBF: {res["rbf_error"]:.3f})',
                linewidth=1.5, markersize=4, color=_color(name))
    ax.set_xlabel('Number of Random Fourier Features')
    ax.set_ylabel('Relative Frobenius Error')
    ax.legend(fontsize=8, frameon=False)
    ax.set_xscale('log')
    ax.axhline(y=0.1, color='#E53935', linestyle='--', alpha=0.6, linewidth=0.8)
    ax.text(60, 0.13, 'Dequantizable threshold (10%)', fontsize=7, color='#E53935', alpha=0.8)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def visualize_kernel_geometry(kernel_matrices, labels, save_path=None):
    _pub_style()
    from sklearn.decomposition import KernelPCA
    from sklearn.manifold import TSNE

    n_kernels = len(kernel_matrices)
    fig, axes = plt.subplots(1, n_kernels, figsize=(3.2 * n_kernels, 3))
    if n_kernels == 1:
        axes = [axes]

    for ax, (name, K) in zip(axes, kernel_matrices.items()):
        try:
            K_sym = (K + K.T) / 2
            min_eig = np.min(np.linalg.eigvalsh(K_sym))
            if min_eig < 0:
                K_sym += (-min_eig + 1e-6) * np.eye(K_sym.shape[0])

            kpca = KernelPCA(n_components=10, kernel='precomputed')
            X_kpca = kpca.fit_transform(K_sym)

            if X_kpca.shape[0] > 5:
                perp = min(30, X_kpca.shape[0] - 1)
                tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
                X_2d = tsne.fit_transform(X_kpca)
            else:
                X_2d = X_kpca[:, :2]

            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                      cmap='coolwarm', alpha=0.6, edgecolors='black',
                      linewidth=0.3, s=25)
            ax.set_title(name, fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)[:50]}", transform=ax.transAxes, ha='center')
            ax.set_title(name, fontsize=9)

    plt.suptitle('Feature Space Geometry (Kernel PCA + t-SNE)', fontsize=11, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()


def generate_summary_report(results, save_path=None):
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT SUMMARY REPORT")
    lines.append("Quantum Kernel for Cross-Subject EEG Emotion Recognition")
    lines.append("=" * 70)

    if 'classification' in results:
        lines.append("\n--- CLASSIFICATION RESULTS (LOSO) ---")
        for task, task_results in results['classification'].items():
            lines.append(f"\nTask: {task}")
            lines.append(f"{'Method':<20} {'Accuracy':>10} {'F1-Macro':>10} {'Std':>10}")
            lines.append("-" * 52)
            for method, metrics in task_results.items():
                acc = metrics.get('mean_accuracy', 0)
                f1 = metrics.get('mean_f1', 0)
                std = metrics.get('std_accuracy', 0)
                lines.append(f"{method:<20} {acc:>10.4f} {f1:>10.4f} {std:>10.4f}")

    if 'kta' in results:
        lines.append("\n--- KERNEL-TARGET ALIGNMENT ---")
        for method, kta_vals in results['kta'].items():
            lines.append(f"  {method}:")
            for task, val in kta_vals.items():
                lines.append(f"    {task}: {val:.4f}")

    if 'dequantization' in results:
        lines.append("\n--- DEQUANTIZATION TEST ---")
        for method, res in results['dequantization'].items():
            lines.append(f"  {method}:")
            lines.append(f"    Best RBF match (relative error): {res['rbf_error']:.4f}")
            if res['rff_errors']:
                min_err = min(res['rff_errors'])
                best_D = res['rff_dims'][res['rff_errors'].index(min_err)]
                lines.append(f"    Best RFF approx: error={min_err:.4f} at D={best_D}")
                lines.append(f"    Dequantizable (error < 10%): {'YES' if min_err < 0.1 else 'NO'}")

    lines.append("\n" + "=" * 70)
    report = "\n".join(lines)
    print(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)

    return report