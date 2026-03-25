import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel


def get_classical_kernels():
    return {
        'SVM-RBF': {'kernel': 'rbf', 'params': {'gamma': 'scale'}},
        'SVM-Poly': {'kernel': 'poly', 'params': {'degree': 3, 'gamma': 'scale', 'coef0': 1}},
        'SVM-Linear': {'kernel': 'linear', 'params': {}},
    }


def train_classical_svm(X_train, y_train, X_test, y_test, kernel_name,
                         C=1.0, return_kernel_matrix=False):
    kernels = get_classical_kernels()
    cfg = kernels[kernel_name]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    svm = SVC(kernel=cfg['kernel'], C=C, class_weight='balanced', **cfg['params'])
    svm.fit(X_tr, y_train)

    y_pred = svm.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    result = {
        'accuracy': acc,
        'f1_macro': f1,
        'y_pred': y_pred,
        'n_support_vectors': svm.n_support_.sum(),
    }

    if return_kernel_matrix:
        X_all = np.vstack([X_tr, X_te])
        if cfg['kernel'] == 'rbf':
            gamma = 1.0 / (X_all.shape[1] * X_all.var())
            K = rbf_kernel(X_all, gamma=gamma)
        elif cfg['kernel'] == 'poly':
            K = polynomial_kernel(X_all, degree=cfg['params']['degree'],
                                  gamma=1.0 / (X_all.shape[1] * X_all.var()),
                                  coef0=cfg['params'].get('coef0', 0))
        else:
            K = X_all @ X_all.T
        result['kernel_matrix'] = K

    return result


def train_quantum_svm(K_train, y_train, K_test, C=1.0):
    svm = SVC(kernel='precomputed', C=C, class_weight='balanced')
    svm.fit(K_train, y_train)
    y_pred = svm.predict(K_test)
    return {'y_pred': y_pred, 'n_support_vectors': svm.n_support_.sum()}


def tune_svm_C(K_train, y_train, C_range=[0.01, 0.1, 1.0, 10.0, 100.0],
               n_inner_folds=3, seed=42):
    from sklearn.model_selection import StratifiedKFold

    n = len(y_train)
    if n < n_inner_folds * 2:
        return 1.0

    skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
    best_C = 1.0
    best_acc = 0.0

    for C in C_range:
        accs = []
        for tr_idx, val_idx in skf.split(np.zeros(n), y_train):
            K_tr = K_train[np.ix_(tr_idx, tr_idx)]
            K_val = K_train[np.ix_(val_idx, tr_idx)]
            try:
                svm = SVC(kernel='precomputed', C=C, class_weight='balanced')
                svm.fit(K_tr, y_train[tr_idx])
                y_pred = svm.predict(K_val)
                accs.append(accuracy_score(y_train[val_idx], y_pred))
            except Exception:
                accs.append(0.5)

        if np.mean(accs) > best_acc:
            best_acc = np.mean(accs)
            best_C = C

    return best_C