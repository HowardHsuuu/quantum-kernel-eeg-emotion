import numpy as np
from scipy import signal
from scipy.signal import hilbert
from config import (
    SAMPLING_RATE, FREQ_BANDS, N_BANDS, BAND_NAMES,
    PSD_NPERSEG, PSD_NOVERLAP, N_EEG_CHANNELS,
    SYMMETRIC_PAIRS, CONNECTIVITY_PAIRS
)


def compute_bandpower(eeg_trial, band_range, fs=SAMPLING_RATE):
    freqs, psd = signal.welch(
        eeg_trial, fs=fs,
        nperseg=min(PSD_NPERSEG, eeg_trial.shape[1]),
        noverlap=min(PSD_NOVERLAP, eeg_trial.shape[1] // 2),
        axis=1
    )
    idx = np.where((freqs >= band_range[0]) & (freqs <= band_range[1]))[0]
    if len(idx) == 0:
        return np.zeros(eeg_trial.shape[0])
    band_power = np.mean(psd[:, idx], axis=1)
    return np.log1p(band_power)


def compute_differential_entropy(eeg_trial, band_range, fs=SAMPLING_RATE):
    low, high = band_range
    nyq = fs / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return np.zeros(eeg_trial.shape[0])
    try:
        b, a = signal.butter(4, [low_n, high_n], btype='band')
        filtered = signal.filtfilt(b, a, eeg_trial, axis=1)
    except ValueError:
        return np.zeros(eeg_trial.shape[0])
    var = np.var(filtered, axis=1)
    var = np.maximum(var, 1e-10)
    return 0.5 * np.log(2 * np.pi * np.e * var)


def compute_dasm(de_features):
    dasm = np.zeros((len(SYMMETRIC_PAIRS), de_features.shape[1]))
    for i, (left_idx, right_idx) in enumerate(SYMMETRIC_PAIRS):
        dasm[i] = de_features[right_idx] - de_features[left_idx]
    return dasm


def compute_plv(eeg_trial, ch_pair, band_range, fs=SAMPLING_RATE):
    low, high = band_range
    nyq = fs / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return 0.0
    try:
        b, a = signal.butter(4, [low_n, high_n], btype='band')
        sig1 = signal.filtfilt(b, a, eeg_trial[ch_pair[0]])
        sig2 = signal.filtfilt(b, a, eeg_trial[ch_pair[1]])
    except ValueError:
        return 0.0
    phase1 = np.angle(hilbert(sig1))
    phase2 = np.angle(hilbert(sig2))
    return np.abs(np.mean(np.exp(1j * (phase1 - phase2))))


def extract_features_trial(eeg_trial):
    n_ch = eeg_trial.shape[0]

    psd = np.zeros((n_ch, N_BANDS))
    for i, (name, band) in enumerate(FREQ_BANDS.items()):
        psd[:, i] = compute_bandpower(eeg_trial, band)

    de = np.zeros((n_ch, N_BANDS))
    for i, (name, band) in enumerate(FREQ_BANDS.items()):
        de[:, i] = compute_differential_entropy(eeg_trial, band)

    dasm = compute_dasm(de)

    plv_alpha = np.array([
        compute_plv(eeg_trial, pair, FREQ_BANDS['alpha'])
        for pair in CONNECTIVITY_PAIRS
    ])
    plv_beta = np.array([
        compute_plv(eeg_trial, pair, FREQ_BANDS['beta'])
        for pair in CONNECTIVITY_PAIRS
    ])

    flat = np.concatenate([
        psd.flatten(),
        de.flatten(),
        dasm.flatten(),
        plv_alpha,
        plv_beta,
    ])

    return {
        'psd': psd, 'de': de, 'dasm': dasm,
        'plv_alpha': plv_alpha, 'plv_beta': plv_beta,
        'flat': flat,
    }


def extract_features_subject(subject_data):
    eeg = subject_data['eeg']
    n_trials = eeg.shape[0]

    all_features = []
    for t in range(n_trials):
        feats = extract_features_trial(eeg[t])
        all_features.append(feats['flat'])

    features = np.array(all_features)

    feature_names = []
    for ch in range(N_EEG_CHANNELS):
        for band in BAND_NAMES:
            feature_names.append(f"PSD_{ch}_{band}")
    for ch in range(N_EEG_CHANNELS):
        for band in BAND_NAMES:
            feature_names.append(f"DE_{ch}_{band}")
    for i, (l, r) in enumerate(SYMMETRIC_PAIRS):
        for band in BAND_NAMES:
            feature_names.append(f"DASM_{i}_{band}")
    for i in range(len(CONNECTIVITY_PAIRS)):
        feature_names.append(f"PLV_alpha_{i}")
    for i in range(len(CONNECTIVITY_PAIRS)):
        feature_names.append(f"PLV_beta_{i}")

    return {
        'features': features,
        'labels_binary': subject_data['labels_binary'],
        'subject_id': subject_data['subject_id'],
        'feature_names': feature_names,
    }


def extract_all_features(subjects, verbose=True):
    all_data = []
    for subj in subjects:
        if verbose:
            print(f"  Extracting features for subject {subj['subject_id']}...")
        feat_data = extract_features_subject(subj)
        all_data.append(feat_data)
        if verbose:
            print(f"    -> {feat_data['features'].shape[1]} features, "
                  f"{feat_data['features'].shape[0]} trials")
    return all_data


def select_features(all_data, task='valence', n_select=20, seed=42):
    from sklearn.feature_selection import mutual_info_classif

    X_all = np.vstack([d['features'] for d in all_data])
    y_all = np.concatenate([d['labels_binary'][task] for d in all_data])
    mi = mutual_info_classif(X_all, y_all, random_state=seed)
    top_idx = np.sort(np.argsort(mi)[-n_select:])

    selected_data = []
    for d in all_data:
        selected_data.append({
            'features': d['features'][:, top_idx],
            'labels_binary': d['labels_binary'],
            'subject_id': d['subject_id'],
            'feature_names': [d['feature_names'][i] for i in top_idx],
        })

    return top_idx, selected_data
