import numpy as np
import pennylane as qml


def _informed_encoding(x, n_qubits, n_layers):
    n = min(len(x), n_qubits)

    for layer in range(n_layers):
        for i in range(n):
            qml.RY(x[i] * np.pi, wires=i)
        for i in range(n, n_qubits):
            qml.Hadamard(wires=i)

        if n_qubits > 4:
            qml.CNOT(wires=[1, 4])
            qml.RZ(x[1] * x[min(4, n - 1)] if n > 4 else 0, wires=4)

        if n_qubits > 3:
            qml.CNOT(wires=[2, 3])
            qml.RZ(x[2] * x[min(3, n - 1)] if n > 3 else 0, wires=3)

        if n_qubits > 1:
            qml.CNOT(wires=[0, 1])

        if n_qubits > 6:
            qml.CNOT(wires=[5, 2])
            qml.CNOT(wires=[6, 3])

        if n_qubits > 7:
            for i in range(5, min(8, n_qubits - 1)):
                qml.CNOT(wires=[i, i + 1])

        if n_qubits > 10:
            qml.CRZ(x[min(9, n - 1)] if n > 9 else 0, wires=[9, 5])
            qml.CRZ(x[min(10, n - 1)] if n > 10 else 0, wires=[10, 6])

        if n_qubits > 11:
            qml.CRZ(x[min(11, n - 1)] if n > 11 else 0, wires=[11, 7])

        if layer < n_layers - 1:
            for i in range(n):
                qml.RX(x[i] * np.pi * 0.5, wires=i)


def _generic_encoding(x, n_qubits, n_layers):
    n = min(len(x), n_qubits)

    for layer in range(n_layers):
        for i in range(n):
            qml.RY(x[i] * np.pi, wires=i)
        for i in range(n, n_qubits):
            qml.Hadamard(wires=i)

        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        if layer < n_layers - 1:
            for i in range(n):
                qml.RX(x[i] * np.pi * 0.5, wires=i)


def _random_encoding(x, n_qubits, n_layers, entangle_pairs, gate_types):
    n = min(len(x), n_qubits)
    gate_map = {'RY': qml.RY, 'RX': qml.RX, 'RZ': qml.RZ}

    for layer in range(n_layers):
        for i in range(n):
            gate_fn = gate_map[gate_types[layer][i]]
            gate_fn(x[i] * np.pi, wires=i)
        for i in range(n, n_qubits):
            qml.Hadamard(wires=i)

        for (a, b) in entangle_pairs[layer]:
            qml.CNOT(wires=[a, b])

        if layer < n_layers - 1:
            for i in range(n):
                qml.RX(x[i] * np.pi * 0.5, wires=i)


def _make_statevector_fn(encoding_fn, n_qubits, **kwargs):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def get_statevector(x):
        encoding_fn(x, n_qubits=n_qubits, **kwargs)
        return qml.state()

    return get_statevector


def create_informed_feature_map(n_qubits, n_layers=2):
    sv_fn = _make_statevector_fn(_informed_encoding, n_qubits, n_layers=n_layers)
    return sv_fn


def create_generic_feature_map(n_qubits, n_layers=2):
    sv_fn = _make_statevector_fn(_generic_encoding, n_qubits, n_layers=n_layers)
    return sv_fn


def create_random_feature_map(n_qubits, n_layers=2, seed=42):
    rng = np.random.RandomState(seed)
    gate_choices = ['RY', 'RX', 'RZ']
    entangle_pairs = []
    gate_types = []
    for _ in range(n_layers):
        pairs = []
        for _ in range(n_qubits - 1):
            a, b = rng.choice(n_qubits, 2, replace=False)
            pairs.append((int(a), int(b)))
        entangle_pairs.append(pairs)
        gate_types.append([gate_choices[rng.randint(3)] for _ in range(n_qubits)])

    sv_fn = _make_statevector_fn(
        _random_encoding, n_qubits,
        n_layers=n_layers, entangle_pairs=entangle_pairs, gate_types=gate_types
    )
    return sv_fn


def _get_statevectors(sv_fn, X, verbose=True):
    n = X.shape[0]
    states = []
    for i in range(n):
        states.append(sv_fn(X[i]))
        if verbose and (i + 1) % 200 == 0:
            print(f"    Statevectors: {i+1}/{n}")
    return np.array(states)


def compute_kernel_matrix(sv_fn, X, Y=None, verbose=True):
    if verbose:
        print(f"    Computing statevectors for X ({X.shape[0]} samples)...")
    states_x = _get_statevectors(sv_fn, X, verbose=verbose)

    if Y is None:
        overlaps = states_x @ states_x.conj().T
        K = np.abs(overlaps) ** 2
        return K
    else:
        if verbose:
            print(f"    Computing statevectors for Y ({Y.shape[0]} samples)...")
        states_y = _get_statevectors(sv_fn, Y, verbose=verbose)
        overlaps = states_x @ states_y.conj().T
        K = np.abs(overlaps) ** 2
        return K


def prepare_features_for_quantum(features, n_qubits=12):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(features)
    n_feat = X_norm.shape[1]
    if n_feat >= n_qubits:
        return X_norm[:, :n_qubits], scaler
    else:
        padding = np.zeros((X_norm.shape[0], n_qubits - n_feat))
        return np.hstack([X_norm, padding]), scaler


def get_all_quantum_kernels(n_qubits=12, n_layers=2, seed=42):
    return {
        'QK-Informed': create_informed_feature_map(n_qubits, n_layers),
        'QK-Generic':  create_generic_feature_map(n_qubits, n_layers),
        'QK-Random':   create_random_feature_map(n_qubits, n_layers, seed),
    }