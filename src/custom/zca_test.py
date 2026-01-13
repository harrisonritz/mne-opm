# %% ZCA test

from turtle import forward
import mne
from preprocessing._config import load_config
from mne_bids import BIDSPath, get_head_mri_trans
from mne.preprocessing.maxwell import _prep_mf_coils, _sss_basis
from mne._fiff.pick import _picks_to_idx, pick_info

import numpy as np
from scipy.linalg import eigh, null_space


import matplotlib.pyplot as plt



# %% parameters

ext_order = 3




# %% get paths

data_path = "/Users/hr0283/Projects/TSX_OPM/data/TSXpilot/bids/sub-013/ses-01/meg/sub-013_ses-01_task-TSXpilot_run-01_split-01_meg.fif"
noise_path = "/Users/hr0283/Projects/TSX_OPM/data/TSXpilot/bids/sub-013/ses-01/meg/sub-013_ses-01_task-noise_meg.fif"

bem_path = "/Users/hr0283/Projects/TSX_OPM/data/TSXpilot/freesurfer/sub-013_ses-01/bem/sub-013_ses-01-5120-bem-sol.fif"
src_path = "/Users/hr0283/Projects/TSX_OPM/data/TSXpilot/freesurfer/sub-013_ses-01/bem/sub-013_ses-01-oct6-src.fif"

trans_path = BIDSPath(
        subject="013",
        session="01",
        task="TSXpilot",
        run="01",
        datatype="meg",
        root="/Users/hr0283/Projects/TSX_OPM/data/TSXpilot/bids")
fs_subjects_dir = "/Users/hr0283/Projects/TSX_OPM/data/TSXpilot/freesurfer"
fs_subject = "sub-013_ses-01"


# %% load data

trans = get_head_mri_trans(
    bids_path=trans_path,
    fs_subjects_dir=fs_subjects_dir,
    fs_subject = fs_subject
)

data = mne.io.read_raw_fif(data_path)
noise = mne.io.read_raw_fif(noise_path)

bem = mne.read_bem_solution(bem_path)
src = mne.read_source_spaces(src_path)

info = data.info

# %% compute noise cov

print("Computing data covariance...")
data_cov = mne.compute_raw_covariance(data, 
                                     method='shrunk', 
                                     rank="info", 
                                     n_jobs=-1)

print("Computing noise covariance...")
noise_cov = mne.compute_raw_covariance(noise, 
                                       method='shrunk', 
                                       rank="info", 
                                       n_jobs=-1)


# %% compute forward

fwd = mne.make_forward_solution(
    info=info, trans=trans, src=src, bem=bem,
    meg=True, eeg=False, mindist=5.0, n_jobs=-1
)

# %% compute inverse

inv = mne.minimum_norm.make_inverse_operator(info, 
                                             fwd, 
                                             noise_cov, 
                                             loose='auto', 
                                             depth=0.8, 
                                             fixed='auto', 
                                             rank="info", 
                                             use_cps=True)

fwd_field = inv['eigen_fields']['data']
fwd_sing = np.diag(inv['sing'])
fwd_leads = inv['eigen_leads']['data']


# plot singular spectrum
plt.figure()
plt.semilogy(np.sort(inv['sing'])[::-1], marker='o')
plt.title("Forward Solution Singular Spectrum")
plt.xlabel("Component")
plt.ylabel("Singular Value (log scale)")
plt.show()
# plot cumulative variance explained
cumulative_variance = np.cumsum(np.sort(inv['sing'])[::-1]**2) / np.sum(inv['sing']**2)
plt.figure()
plt.plot(cumulative_variance, marker='o')
plt.title("Cumulative Variance Explained by Forward Solution Components")   
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.ylim(0, 1)
plt.grid()
plt.show()



# %% compute external SSS basis

sss_info = pick_info(info, mne.pick_types(info, meg=True, eeg=False, exclude='bads'))
exp = dict(origin=(0.0, 0.0, 0.0), int_order=0, ext_order=ext_order)
coils = _prep_mf_coils(sss_info, ignore_ref=True, accuracy='accurate')
n_chs = len(coils[5])

if n_chs != sss_info["nchan"]:
    raise ValueError(
        f"Only {n_chs}/{sss_info['nchan']} picks could be interpreted as MEG channels."
    )
ext_basis = _sss_basis(exp, coils)


# %% compute signal and noise transforms

# wt_fwd = fwd_field @ fwd_sing @ fwd_leads.T
# get column norms of wt_fwd
# col_norms = np.linalg.norm(wt_fwd, axis=0)
# signal_trans = wt_fwd @ wt_fwd.T

signal_trans = fwd_field @ (fwd_sing**2) @ fwd_field.T
noise_trans = ext_basis @ ext_basis.T

# plot signal_trans and noise_trans heatmaps to compare
# match scale of colorbars
vmin = min(signal_trans.min(), noise_trans.min())
vmax = max(signal_trans.max(), noise_trans.max())
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Signal Transform")
plt.imshow(signal_trans, cmap='viridis')
plt.colorbar()
plt.clim(vmin, vmax)

plt.subplot(1, 2, 2)
plt.title("Noise Transform")
plt.imshow(noise_trans, cmap='viridis')
plt.colorbar()
plt.clim(vmin, vmax)

plt.show()


# %% compute GED
reg = 1e-6

signal_cov = signal_trans @ data_cov['data'] @ signal_trans.T
noise_cov = noise_trans @ data_cov['data'] @ noise_trans.T
noise_cov = (1-reg)*noise_cov + reg * np.trace(noise_cov) / noise_cov.shape[0] * np.eye(noise_cov.shape[0])

# Symmetrize (numerical hygiene)
signal_cov = (signal_cov + signal_cov.T) / 2
noise_cov = (noise_cov + noise_cov.T) / 2

# Generalized eigendecomposition
eigenvalues, eigenvectors = eigh(signal_cov, noise_cov)


ged = 


# plot signal_trans and noise_trans heatmaps to compare
# match scale of colorbars
vmin = min(signal_cov.min(), noise_cov.min())
vmax = max(signal_cov.max(), noise_cov.max())
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Signal Covariance")
plt.imshow(signal_cov, cmap='viridis')
plt.colorbar()
plt.clim(vmin, vmax)

plt.subplot(1, 2, 2)
plt.title("Noise Covariance")
plt.imshow(noise_cov, cmap='viridis')
plt.colorbar()
plt.clim(vmin, vmax)

plt.show()


# plot eigenspectrum
plt.figure()
plt.semilogy(np.sort(eigenvalues)[::-1], marker='o')
plt.title("GED Eigenspectrum")
plt.xlabel("Component")
plt.ylabel("Eigenvalue (log scale)")
plt.show()
# plot top 5 GED components
n_components = 5
fig, axes = plt.subplots(1, n_components, figsize=(15, 5))
for i in range(n_components):
    comp = eigenvectors[:, -1 - i]
    im = axes[i].imshow(np.outer(comp, comp), cmap='viridis')
    axes[i].set_title(f"GED Component {i+1}")
    plt.colorbar(im, ax=axes[i])
plt.show()

# plot cumulative variance explained
cumulative_variance = np.cumsum(np.sort(eigenvalues)[::-1]) / np.sum(eigenvalues)
plt.figure()
plt.plot(cumulative_variance, marker='o')
plt.title("Cumulative Variance Explained by GED Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.ylim(0, 1)
plt.grid()
plt.show()








# %%
