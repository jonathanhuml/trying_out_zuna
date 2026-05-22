# !pip install -e git+ssh://git@github.com/jonathanhuml/braindecode.git@zuna#egg=braindecode
# !pip install -e git+https://github.com/neurotechx/moabb.git#egg=moabb
# !pip install -e 

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
os.environ["MNE_DATA"] = str(DATA_DIR)
os.environ["MNE_DATASETS_BNCI_PATH"] = str(DATA_DIR)

import mne

mne.set_config("MNE_DATA", str(DATA_DIR), set_env=True)
mne.set_config("MNE_DATASETS_BNCI_PATH", str(DATA_DIR), set_env=True)

##### IMPORT DATA #####

from moabb.datasets import BNCI2015_001
from moabb.paradigms import MotorImagery
fmin = 0.5
fmax = 45
BASELINE_SFREQ = 128
ZUNA_SFREQ = 256

dataset = BNCI2015_001()
paradigm = MotorImagery(
    fmin=fmin, fmax=fmax, resample=BASELINE_SFREQ
)
zuna_paradigm = MotorImagery(
    fmin=fmin, fmax=fmax, resample=ZUNA_SFREQ
)

data = paradigm.get_data(dataset=dataset)
X_data, y_data, metadata = data
expected_trial_duration_s = 5.0
dataset_trial_duration_s = dataset.interval[1] - dataset.interval[0]
sfreq = BASELINE_SFREQ
sample_trial_duration_s = X_data.shape[-1] / sfreq

print(
    "Trial duration check: "
    f"dataset_interval={dataset_trial_duration_s:.3f}s, "
    f"samples={X_data.shape[-1]}, sfreq={sfreq:.1f}Hz, "
    f"sample_duration={sample_trial_duration_s:.3f}s"
)
##### SEED SETTING + IMPORTS #####

import random
import numpy as np
import torch
from braindecode.util import set_random_seeds
# Enable synchronous CUDA error reporting for easier debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 1) Set all seeds for Python, NumPy, and Torch
set_random_seeds(42, cuda=True)

# 2) Configure PyTorch for deterministic operations
# Turn on full determinism if possible:
# torch.use_deterministic_algorithms(True)  # safer, but can error if certain ops don't have deterministic backends
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

##### PIPELINE #####

import matplotlib.pyplot as plt
import mne
import pandas as pd
import seaborn as sns
import torch
from braindecode import EEGClassifier
from braindecode.models import EEGNeX, BrainModule, DGCNN, ZUNA

from sklearn.pipeline import make_pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit

from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.utils import setup_seed as moabb_setup_seed

# Ensure MOABB also uses the seed

mne.set_log_level(False)

# Print Information PyTorch
print(f"Torch Version: {torch.__version__}")

# Set up GPU if it is there
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
print("GPU is", "AVAILABLE" if cuda else "NOT AVAILABLE")

# Set random seed to be able to reproduce results
seed = 42
moabb_setup_seed(seed)


# Hyperparameter
LEARNING_RATE = 1E-3 # parameter taken from Braindecode
WEIGHT_DECAY = 0.01  # parameter taken from Braindecode
BATCH_SIZE = 64  # parameter taken from BrainDecode
EPOCH = 100
PATIENCE = 50
sfreq = BASELINE_SFREQ

# Dynamically determine the number of classes from the loaded data
# data[1] contains the labels from paradigm.get_data
n_classes = len(np.unique(data[1]))
subjects = [1]
X, _, _ = paradigm.get_data(dataset=dataset, subjects=subjects)
n_times = X.shape[2]
n_chans = X.shape[1]
info = mne.create_info(dataset.METADATA.acquisition.sensors, sfreq=sfreq, ch_types="eeg")
info.set_montage(mne.channels.make_standard_montage("standard_1020"))
chs_info = info["chs"]
zuna_info = mne.create_info(
    dataset.METADATA.acquisition.sensors, sfreq=ZUNA_SFREQ, ch_types="eeg"
)
zuna_info.set_montage(mne.channels.make_standard_montage("standard_1020"))
zuna_chs_info = zuna_info["chs"]
zuna_X, _, _ = zuna_paradigm.get_data(dataset=dataset, subjects=subjects)
zuna_n_times = zuna_X.shape[2]

##### CREATE MODEL #####

def create_model(model_name, n_times, n_chans, n_outputs, class_module, sfreq):

    return EEGClassifier(
    module=class_module,
    optimizer=torch.optim.AdamW,
    optimizer__lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=EPOCH,
    train_split=ValidSplit(0.2, random_state=seed, stratified=True),
    device=device,
    callbacks=[
        EarlyStopping(monitor="valid_loss", patience=PATIENCE),
        EpochScoring(
            scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
        ),
        EpochScoring(
            scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
        ),
    ],
    verbose=1,
    criterion=torch.nn.CrossEntropyLoss,
)

##### PIPELINE PER MODEL #####
# Create the pipelines
model_configs = [
    ("EEGNeX", EEGNeX(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)),
    ("BrainModule", BrainModule(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)),
    ("DGCNN", DGCNN(chs_info=chs_info, n_outputs=n_classes, n_times=n_times)),
    ("ZUNA", ZUNA(chs_info=zuna_chs_info, n_outputs=n_classes, n_times=zuna_n_times, sfreq=ZUNA_SFREQ)),
]

pipes = {}
for model_name, class_module in model_configs:
    model_n_times = zuna_n_times if model_name == "ZUNA" else n_times
    model_sfreq = ZUNA_SFREQ if model_name == "ZUNA" else sfreq
    pipes[model_name] = make_pipeline(
        create_model(model_name, model_n_times, n_chans, n_classes, class_module, sfreq=model_sfreq)
    )


evaluation = CrossSessionEvaluation(
    paradigm=paradigm,
    datasets=dataset,
    suffix="example",
    overwrite=True,
    n_jobs=1,
    random_state=seed,
)

results = evaluation.process({k: v for k, v in pipes.items() if k != "ZUNA"})
zuna_results = CrossSessionEvaluation(
    paradigm=zuna_paradigm,
    datasets=dataset,
    suffix="zuna",
    overwrite=True,
    n_jobs=1,
    random_state=seed,
).process({"ZUNA": pipes["ZUNA"]})
results = pd.concat([results, zuna_results], ignore_index=True)

print(results.head())

def around(x):
  mean = np.around(x*100, 2)
  return mean

pd.options.display.float_format = '{:.2f}'.format
mean = results.groupby(["pipeline", "session", "subject"])["score"].agg([around]).unstack()

mean

fig, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data=results, y="score", x="subject", hue="pipeline", ax=ax)
ax.legend(loc=2, borderaxespad=0., ncols=5)
plt.ylim(0.0, 1.0)

plt.show()

plt.figure()
sns.barplot(data=results, y="score", x="pipeline", palette="viridis")
plt.show()
