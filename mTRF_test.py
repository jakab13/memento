from os.path import join
import pathlib
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
import mne
from mne.decoding import ReceptiveField

DIR = pathlib.Path(os.getcwd())

path = DIR / "data" / "mne_data" / "mTRF_1.5"
decim = 2
data = loadmat(join(path, "speech_data.mat"))
raw = data["EEG"].T
speech = data["envelope"].T
sfreq = float(data["Fs"].item())
sfreq /= decim
speech = mne.filter.resample(speech, down=decim)
raw = mne.filter.resample(raw, down=decim)

# Read in channel positions and create our MNE objects from the raw data
montage = mne.channels.make_standard_montage("biosemi128")
info = mne.create_info(montage.ch_names, sfreq, "eeg").set_montage(montage)
raw = mne.io.RawArray(raw, info)
n_channels = len(raw.ch_names)

# Plot a sample of brain and stimulus activity
fig, ax = plt.subplots()
lns = ax.plot(scale(raw[:, :800][0].T), color="k", alpha=0.1)
ln1 = ax.plot(scale(speech[0, :800]), color="r", lw=2)
ax.legend([lns[0], ln1[0]], ["EEG", "Speech Envelope"], frameon=False)
ax.set(title="Sample activity", xlabel="Time (s)")

# Define the delays that we will use in the receptive field
tmin, tmax = -0.2, 0.4

# Initialize the model
rf = ReceptiveField(
    tmin, tmax, sfreq, feature_names=["envelope"], estimator=1.0, scoring="corrcoef"
)
# We'll have (tmax - tmin) * sfreq delays
# and an extra 2 delays since we are inclusive on the beginning / end index
n_delays = int((tmax - tmin) * sfreq) + 2

n_splits = 3
cv = KFold(n_splits)

# Prepare model data (make time the first dimension)
speech = speech.T
Y, _ = raw[:]  # Outputs for the model
Y = Y.T

# Iterate through splits, fit the model, and predict/test on held-out data
coefs = np.zeros((n_splits, n_channels, n_delays))
scores = np.zeros((n_splits, n_channels))
for ii, (train, test) in enumerate(cv.split(speech)):
    print(f"split {ii + 1} / {n_splits}")
    rf.fit(speech[train], Y[train])
    scores[ii] = rf.score(speech[test], Y[test])
    # coef_ is shape (n_outputs, n_features, n_delays). we only have 1 feature
    coefs[ii] = rf.coef_[:, 0, :]
times = rf.delays_ / float(rf.sfreq)

# Average scores and coefficients across CV splits
mean_coefs = coefs.mean(axis=0)
mean_scores = scores.mean(axis=0)

# Plot mean prediction scores across all channels
fig, ax = plt.subplots(layout="constrained")
ix_chs = np.arange(n_channels)
ax.plot(ix_chs, mean_scores)
ax.axhline(0, ls="--", color="r")
ax.set(title="Mean prediction score", xlabel="Channel", ylabel="Score ($r$)")

# Print mean coefficients across all time delays / channels (see Fig 1)
time_plot = 0.180  # For highlighting a specific time.
fig, ax = plt.subplots(figsize=(4, 8), layout="constrained")
max_coef = mean_coefs.max()
ax.pcolormesh(
    times,
    ix_chs,
    mean_coefs,
    cmap="RdBu_r",
    vmin=-max_coef,
    vmax=max_coef,
    shading="gouraud",
)
ax.axvline(time_plot, ls="--", color="k", lw=2)
ax.set(
    xlabel="Delay (s)",
    ylabel="Channel",
    title="Mean Model\nCoefficients",
    xlim=times[[0, -1]],
    ylim=[len(ix_chs) - 1, 0],
    xticks=np.arange(tmin, tmax + 0.2, 0.2),
)
plt.setp(ax.get_xticklabels(), rotation=45)

# Make a topographic map of coefficients for a given delay (see Fig 2C)
ix_plot = np.argmin(np.abs(time_plot - times))
fig, ax = plt.subplots(layout="constrained")
mne.viz.plot_topomap(
    mean_coefs[:, ix_plot], pos=info, axes=ax, show=False, vlim=(-max_coef, max_coef)
)
ax.set(title="Topomap of model coefficients\nfor delay %s" % time_plot)

# We use the same lags as in :footcite:`CrosseEtAl2016`. Negative lags now
# index the relationship
# between the neural response and the speech envelope earlier in time, whereas
# positive lags would index how a unit change in the amplitude of the EEG would
# affect later stimulus activity (obviously this should have an amplitude of
# zero).
tmin, tmax = -0.2, 0.0

# Initialize the model. Here the features are the EEG data. We also specify
# ``patterns=True`` to compute inverse-transformed coefficients during model
# fitting (cf. next section and :footcite:`HaufeEtAl2014`).
# We'll use a ridge regression estimator with an alpha value similar to
# Crosse et al.
sr = ReceptiveField(
    tmin,
    tmax,
    sfreq,
    feature_names=raw.ch_names,
    estimator=1e4,
    scoring="corrcoef",
    patterns=True,
)
# We'll have (tmax - tmin) * sfreq delays
# and an extra 2 delays since we are inclusive on the beginning / end index
n_delays = int((tmax - tmin) * sfreq) + 2

n_splits = 3
cv = KFold(n_splits)

# Iterate through splits, fit the model, and predict/test on held-out data
coefs = np.zeros((n_splits, n_channels, n_delays))
patterns = coefs.copy()
scores = np.zeros((n_splits,))
for ii, (train, test) in enumerate(cv.split(speech)):
    print(f"split {ii + 1} / {n_splits}")
    sr.fit(Y[train], speech[train])
    scores[ii] = sr.score(Y[test], speech[test])[0]
    # coef_ is shape (n_outputs, n_features, n_delays). We have 128 features
    coefs[ii] = sr.coef_[0, :, :]
    patterns[ii] = sr.patterns_[0, :, :]
times = sr.delays_ / float(sr.sfreq)

# Average scores and coefficients across CV splits
mean_coefs = coefs.mean(axis=0)
mean_patterns = patterns.mean(axis=0)
mean_scores = scores.mean(axis=0)
max_coef = np.abs(mean_coefs).max()
max_patterns = np.abs(mean_patterns).max()

y_pred = sr.predict(Y[test])
time = np.linspace(0, 2.0, 5 * int(sfreq))
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
ax.plot(
    time, speech[test][sr.valid_samples_][: int(5 * sfreq)], color="grey", lw=2, ls="--"
)
ax.plot(time, y_pred[sr.valid_samples_][: int(5 * sfreq)], color="r", lw=2)
ax.legend([lns[0], ln1[0]], ["Envelope", "Reconstruction"], frameon=False)
ax.set(title="Stimulus reconstruction")
ax.set_xlabel("Time (s)")

time_plot = (-0.140, -0.125)  # To average between two timepoints.
ix_plot = np.arange(
    np.argmin(np.abs(time_plot[0] - times)), np.argmin(np.abs(time_plot[1] - times))
)
fig, ax = plt.subplots(1, 2)
mne.viz.plot_topomap(
    np.mean(mean_coefs[:, ix_plot], axis=1),
    pos=info,
    axes=ax[0],
    show=False,
    vlim=(-max_coef, max_coef),
)
ax[0].set(title=f"Model coefficients\nbetween delays {time_plot[0]} and {time_plot[1]}")

mne.viz.plot_topomap(
    np.mean(mean_patterns[:, ix_plot], axis=1),
    pos=info,
    axes=ax[1],
    show=False,
    vlim=(-max_patterns, max_patterns),
)
ax[1].set(
    title=(
        f"Inverse-transformed coefficients\nbetween delays {time_plot[0]} and "
        f"{time_plot[1]}"
    )
)