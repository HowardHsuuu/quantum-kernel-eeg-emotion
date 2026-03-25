import os
import pickle
import numpy as np
from config import (
    DEAP_DIR, N_SUBJECTS, N_EEG_CHANNELS,
    N_SAMPLES, LABEL_NAMES
)

THRESHOLD = 5.0
MIN_MINORITY_RATIO = float(os.environ.get("MIN_MINORITY_RATIO", "0.15"))


def _load_raw(subject_id):
    fname = f"s{subject_id:02d}.dat"
    fpath = os.path.join(DEAP_DIR, fname)

    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"DEAP file not found: {fpath}\n"
            f"Place s01.dat - s32.dat in {DEAP_DIR}/ or set DEAP_DIR env var."
        )

    with open(fpath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return {
        'eeg': data['data'][:, :N_EEG_CHANNELS, :],
        'labels_raw': data['labels'],
        'subject_id': subject_id,
    }


def load_all_subjects(subject_ids=None):
    if subject_ids is None:
        subject_ids = list(range(1, N_SUBJECTS + 1))

    raw_subjects = []
    for sid in subject_ids:
        try:
            raw_subjects.append(_load_raw(sid))
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    if len(raw_subjects) == 0:
        return []

    subjects = []
    skipped = []
    for s in raw_subjects:
        labels_binary = {}
        for idx, name in LABEL_NAMES.items():
            labels_binary[name] = (s['labels_raw'][:, idx] > THRESHOLD).astype(int)
        s['labels_binary'] = labels_binary

        sid = s['subject_id']
        vbal = labels_binary['valence'].mean()
        abal = labels_binary['arousal'].mean()

        min_ratio = min(vbal, 1 - vbal, abal, 1 - abal)
        if min_ratio < MIN_MINORITY_RATIO:
            skipped.append(sid)
            print(f"  Skipped s{sid:02d}: v={vbal:.2f} a={abal:.2f} (too imbalanced)")
            continue

        subjects.append(s)
        print(f"  Loaded s{sid:02d}: {s['eeg'].shape}, v={vbal:.2f} a={abal:.2f}")

    print(f"\nLoaded {len(subjects)} / {len(subject_ids)} subjects "
          f"(skipped {len(skipped)} for imbalance)")
    return subjects


def generate_synthetic_deap(n_subjects=8, n_trials=40, seed=42):
    rng = np.random.RandomState(seed)
    subjects = []

    for sid in range(1, n_subjects + 1):
        subj_bias = rng.randn(N_EEG_CHANNELS) * 0.3
        eeg = np.zeros((n_trials, N_EEG_CHANNELS, N_SAMPLES))
        labels_raw = np.zeros((n_trials, 4))

        for t in range(n_trials):
            valence = rng.uniform(1, 9)
            arousal = rng.uniform(1, 9)
            labels_raw[t] = [valence, arousal, rng.uniform(1, 9), rng.uniform(1, 9)]
            for ch in range(N_EEG_CHANNELS):
                time = np.arange(N_SAMPLES) / 128.0
                alpha_amp = 1.0 + 0.3 * (valence - 5) / 4 * (1 if ch < 16 else -1)
                beta_amp = 0.5 + 0.2 * (arousal - 5) / 4
                eeg[t, ch] = (
                    0.8 * np.sin(2 * np.pi * 6 * time) +
                    alpha_amp * np.sin(2 * np.pi * 10 * time) +
                    beta_amp * np.sin(2 * np.pi * 22 * time) +
                    rng.randn(N_SAMPLES) * 0.5
                ) * (1 + subj_bias[ch])

        labels_binary = {
            name: (labels_raw[:, idx] > THRESHOLD).astype(int)
            for idx, name in LABEL_NAMES.items()
        }
        subjects.append({
            'eeg': eeg, 'labels_raw': labels_raw,
            'labels_binary': labels_binary, 'subject_id': sid,
        })

    print(f"Generated {n_subjects} synthetic subjects (shape: {eeg.shape})")
    return subjects