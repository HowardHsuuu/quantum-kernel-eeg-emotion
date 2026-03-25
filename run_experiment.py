import argparse
import time
import json
import os
import numpy as np
from collections import defaultdict

from config import (
    setup_dirs, OUTPUT_DIR, FIGURES_DIR, RANDOM_SEED,
    N_QUBITS_TOTAL, N_LAYERS, N_FEATURES_SELECT, SVM_C_RANGE
)
from data_loader import load_all_subjects, generate_synthetic_deap
from features import extract_all_features, select_features
from quantum_kernels import (
    get_all_quantum_kernels, compute_kernel_matrix,
    prepare_features_for_quantum
)
from classical_baselines import (
    get_classical_kernels, train_classical_svm,
    train_quantum_svm, tune_svm_C
)
from analysis import (
    analyze_eigenspectrum, compute_kta, compute_centered_kta,
    plot_kta_comparison, rff_approximate_kernel, plot_dequantization,
    visualize_kernel_geometry, generate_summary_report
)
from sklearn.metrics import accuracy_score, f1_score


def run_loso_experiment(all_data, task='valence', n_qubits=12, n_layers=2,
                        n_features=20):
    n_subjects = len(all_data)
    print(f"\n{'='*60}")
    print(f"LOSO Experiment: {task} ({n_subjects} subjects)")
    print(f"Qubits: {n_qubits}, Layers: {n_layers}, Features: {n_features}")
    print(f"{'='*60}")

    print("\n[1/5] Initializing quantum kernels...")
    qk_dict = get_all_quantum_kernels(n_qubits=n_qubits, n_layers=n_layers)
    classical_names = list(get_classical_kernels().keys())

    results = {
        'classification': defaultdict(lambda: defaultdict(list)),
        'kta': defaultdict(lambda: defaultdict(list)),
        'kernel_matrices': {},
    }

    for fold, test_subj_idx in enumerate(range(n_subjects)):
        test_subj = all_data[test_subj_idx]
        train_subjs = [all_data[i] for i in range(n_subjects) if i != test_subj_idx]

        from sklearn.feature_selection import mutual_info_classif
        X_pool = np.vstack([s['features'] for s in train_subjs])
        y_pool = np.concatenate([s['labels_binary'][task] for s in train_subjs])
        mi = mutual_info_classif(X_pool, y_pool, random_state=42)
        selected_indices = np.sort(np.argsort(mi)[-n_features:])

        X_train = X_pool[:, selected_indices]
        y_train = y_pool
        X_test = test_subj['features'][:, selected_indices]
        y_test = test_subj['labels_binary'][task]

        sid = test_subj['subject_id']
        print(f"\n--- Fold {fold+1}/{n_subjects}: test=s{sid:02d} "
              f"(train={len(y_train)}, test={len(y_test)}) ---")

        if len(np.unique(y_test)) < 2:
            print(f"  [SKIP] Only one class in test set")
            continue

        for cl_name in classical_names:
            store_km = (fold == n_subjects - 1) and (len(y_train) + len(y_test) <= 1500)
            res = train_classical_svm(X_train, y_train, X_test, y_test,
                                      cl_name, C=1.0,
                                      return_kernel_matrix=store_km)
            results['classification'][cl_name]['accuracies'].append(res['accuracy'])
            results['classification'][cl_name]['f1s'].append(res['f1_macro'])
            print(f"  {cl_name}: acc={res['accuracy']:.3f}")
            if 'kernel_matrix' in res:
                results['kernel_matrices'][cl_name] = res['kernel_matrix']

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        def _prep_q(X_scaled, n_q):
            if X_scaled.shape[1] >= n_q:
                return X_scaled[:, :n_q]
            pad = np.zeros((X_scaled.shape[0], n_q - X_scaled.shape[1]))
            return np.hstack([X_scaled, pad])

        X_q_train = _prep_q(X_train_scaled, n_qubits)
        X_q_test = _prep_q(X_test_scaled, n_qubits)

        for qk_name, kernel_circuit in qk_dict.items():
            print(f"  Computing {qk_name} kernel...")
            t0 = time.time()

            K_train = compute_kernel_matrix(kernel_circuit, X_q_train, verbose=False)
            K_test = compute_kernel_matrix(kernel_circuit, X_q_test, X_q_train, verbose=False)
            print(f"    Done in {time.time() - t0:.1f}s")

            best_C = tune_svm_C(K_train, y_train, C_range=SVM_C_RANGE)
            qk_res = train_quantum_svm(K_train, y_train, K_test, C=best_C)

            acc = accuracy_score(y_test, qk_res['y_pred'])
            f1 = f1_score(y_test, qk_res['y_pred'], average='macro', zero_division=0)
            results['classification'][qk_name]['accuracies'].append(acc)
            results['classification'][qk_name]['f1s'].append(f1)
            print(f"  {qk_name}: acc={acc:.3f} (C={best_C})")

            kta = compute_kta(K_train, y_train)
            results['kta'][qk_name][task].append(kta)

            if fold == n_subjects - 1 and len(y_train) + len(y_test) <= 1500:
                X_q_all = np.vstack([X_q_train, X_q_test])
                K_full = compute_kernel_matrix(kernel_circuit, X_q_all, verbose=False)
                results['kernel_matrices'][qk_name] = K_full

    return results


def aggregate_results(results):
    summary = {'classification': {}, 'kta': {}}
    for method, metrics in results['classification'].items():
        accs = metrics['accuracies']
        f1s = metrics['f1s']
        summary['classification'][method] = {
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
            'mean_f1': np.mean(f1s),
            'std_f1': np.std(f1s),
            'n_folds': len(accs),
            'per_fold_accuracy': accs,
            'per_fold_f1': f1s,
        }
    for method, kta_dict in results['kta'].items():
        summary['kta'][method] = {}
        for t, vals in kta_dict.items():
            summary['kta'][method][t] = np.mean(vals)
    return summary


def run_analysis(results, all_data, task, n_qubits):
    print(f"\n{'='*60}")
    print("POST-HOC ANALYSIS")
    print(f"{'='*60}")

    analysis_results = {}

    if results.get('kernel_matrices'):
        print("\n[A1] Eigenspectrum analysis...")
        eigenvals = analyze_eigenspectrum(
            results['kernel_matrices'],
            save_path=os.path.join(FIGURES_DIR, f'eigenspectrum_{task}.png')
        )
        analysis_results['eigenspectrum'] = {
            name: {'effective_rank': float(np.exp(-np.sum(
                (v[v > 0] / v[v > 0].sum()) * np.log(v[v > 0] / v[v > 0].sum() + 1e-15)
            )))}
            for name, v in eigenvals.items()
        }

    if results.get('kernel_matrices'):
        print("\n[A2] Feature space geometry...")
        y_all = np.concatenate([s['labels_binary'][task] for s in all_data])
        for name, K in results['kernel_matrices'].items():
            y_vis = y_all[:K.shape[0]]
            break
        visualize_kernel_geometry(
            results['kernel_matrices'], y_vis,
            save_path=os.path.join(FIGURES_DIR, f'geometry_{task}.png')
        )

    if results.get('kernel_matrices'):
        print("\n[A3] RFF dequantization test...")
        X_all = np.vstack([s['features'] for s in all_data])
        first_K = results['kernel_matrices'][list(results['kernel_matrices'].keys())[0]]
        X_q, _ = prepare_features_for_quantum(X_all[:first_K.shape[0]], n_qubits)

        rff_results = {}
        for name, K in results['kernel_matrices'].items():
            if name.startswith('QK'):
                print(f"  Testing {name}...")
                rff_res = rff_approximate_kernel(K, X_q)
                rff_results[name] = rff_res
                min_err = min(rff_res['rff_errors'])
                print(f"    Best RBF match: {rff_res['rbf_error']:.4f}")
                print(f"    Best RFF error: {min_err:.4f}")
                print(f"    Dequantizable: {'YES' if min_err < 0.1 else 'NO'}")

        if rff_results:
            plot_dequantization(
                rff_results,
                save_path=os.path.join(FIGURES_DIR, f'dequantization_{task}.png')
            )
            analysis_results['dequantization'] = rff_results

    return analysis_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--task', default='valence', choices=['valence', 'arousal'])
    parser.add_argument('--n-qubits', type=int, default=N_QUBITS_TOTAL)
    parser.add_argument('--n-layers', type=int, default=N_LAYERS)
    parser.add_argument('--n-features', type=int, default=N_FEATURES_SELECT)
    args = parser.parse_args()

    setup_dirs()
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("QUANTUM KERNEL FOR CROSS-SUBJECT EEG EMOTION RECOGNITION")
    print("=" * 60)

    if args.synthetic or args.quick:
        n_subj = 4 if args.quick else 8
        n_trials = 20 if args.quick else 40
        print(f"\n[DATA] Generating synthetic data ({n_subj} subjects)...")
        subjects = generate_synthetic_deap(n_subjects=n_subj, n_trials=n_trials)
    else:
        print(f"\n[DATA] Loading DEAP dataset...")
        subjects = load_all_subjects()
        if len(subjects) == 0:
            print("No DEAP data found. Use --synthetic for testing.")
            return

    print(f"\n[FEATURES] Extracting features...")
    all_data = extract_all_features(subjects)

    n_qubits = 6 if args.quick else args.n_qubits
    n_layers = 1 if args.quick else args.n_layers
    n_features = 10 if args.quick else args.n_features

    t_start = time.time()
    results = run_loso_experiment(
        all_data, task=args.task,
        n_qubits=n_qubits, n_layers=n_layers,
        n_features=n_features
    )
    t_exp = time.time() - t_start

    summary = aggregate_results(results)
    analysis_results = run_analysis(results, all_data, args.task, n_qubits)

    from scipy.stats import wilcoxon
    stat_results = {}
    methods = summary['classification']
    qk_names = [m for m in methods if m.startswith('QK')]
    cl_names = [m for m in methods if not m.startswith('QK')]

    print(f"\n{'='*60}")
    print("STATISTICAL TESTS (Wilcoxon signed-rank, one-sided)")
    print(f"{'='*60}")

    for qk in qk_names:
        for cl in cl_names:
            q = np.array(methods[qk]['per_fold_accuracy'])
            c = np.array(methods[cl]['per_fold_accuracy'])
            diff = q - c
            if np.all(diff == 0):
                print(f"  {qk} vs {cl}: identical")
                continue
            try:
                stat, p = wilcoxon(q, c, alternative='greater')
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {qk} vs {cl}: p={p:.4f} {sig} (mean diff={diff.mean():+.4f})")
                stat_results[f"{qk}_vs_{cl}"] = {'p': p, 'mean_diff': float(diff.mean())}
            except Exception as e:
                print(f"  {qk} vs {cl}: error ({e})")

    for qk1, qk2 in [('QK-Informed','QK-Generic'),('QK-Informed','QK-Random'),('QK-Generic','QK-Random')]:
        if qk1 in methods and qk2 in methods:
            q1 = np.array(methods[qk1]['per_fold_accuracy'])
            q2 = np.array(methods[qk2]['per_fold_accuracy'])
            diff = q1 - q2
            if np.all(diff == 0):
                print(f"  {qk1} vs {qk2}: identical")
                continue
            try:
                stat, p = wilcoxon(q1, q2, alternative='greater')
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {qk1} vs {qk2}: p={p:.4f} {sig} (mean diff={diff.mean():+.4f})")
                stat_results[f"{qk1}_vs_{qk2}"] = {'p': p, 'mean_diff': float(diff.mean())}
            except Exception as e:
                print(f"  {qk1} vs {qk2}: error ({e})")

    full_results = {
        'classification': {args.task: summary['classification']},
        'kta': summary['kta'],
        'dequantization': analysis_results.get('dequantization', {}),
        'statistical_tests': stat_results,
        'config': {
            'task': args.task,
            'n_subjects': len(subjects),
            'n_qubits': n_qubits,
            'n_layers': n_layers,
            'n_features': n_features,
            'synthetic': args.synthetic or args.quick,
            'total_time_seconds': t_exp,
        },
    }

    generate_summary_report(
        full_results,
        save_path=os.path.join(OUTPUT_DIR, f'report_{args.task}.txt')
    )

    def convert(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, defaultdict): return dict(obj)
        return obj

    with open(os.path.join(OUTPUT_DIR, f'results_{args.task}.json'), 'w') as f:
        json.dump(json.loads(json.dumps(full_results, default=convert)), f, indent=2)

    print(f"\nDone. Total time: {t_exp/60:.1f} min")
    print(f"Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()