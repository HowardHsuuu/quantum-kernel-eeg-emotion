import os

DEAP_DIR = os.environ.get("DEAP_DIR", "./data/deap")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

N_SUBJECTS = 32
N_TRIALS = 40
N_EEG_CHANNELS = 32
N_ALL_CHANNELS = 40
N_SAMPLES = 8064
SAMPLING_RATE = 128

CHANNEL_NAMES = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
    'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
    'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2',
]

SYMMETRIC_PAIRS = [
    (0, 16), (1, 17), (2, 19), (3, 20), (4, 21), (5, 22), (6, 24),
    (7, 25), (8, 26), (9, 27), (10, 28), (11, 29), (12, 30), (13, 31),
]

FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta':  (14, 30),
    'gamma': (30, 45),
}
N_BANDS = len(FREQ_BANDS)
BAND_NAMES = list(FREQ_BANDS.keys())

PSD_NPERSEG = 256
PSD_NOVERLAP = 128

CONNECTIVITY_PAIRS = [
    (0, 10), (16, 28), (2, 10), (19, 28),
    (1, 12), (17, 30), (6, 10), (24, 28),
]

LABEL_THRESHOLD = 5.0
LABEL_NAMES = {0: 'valence', 1: 'arousal'}

N_QUBITS_TOTAL = 12
N_LAYERS = 2
N_FEATURES_SELECT = 20

SVM_C_RANGE = [0.01, 0.1, 1.0, 10.0, 100.0]
RANDOM_SEED = 42
N_RFF_DIMS = [50, 100, 200, 500, 1000, 2000]


def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DEAP_DIR, exist_ok=True)
