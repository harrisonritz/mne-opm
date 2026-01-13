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
fwd_sing = inv['sing']
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
ext_basis /= (np.linalg.norm(ext_basis, axis=0)**0.8)


# %% compute signal and noise transforms

# fwd_full = fwd_field @ np.diag(fwd_sing) @ fwd_leads.T
# fwd_full /= np.linalg.norm(fwd_full, axis=0)

# get column norms of wt_fwd
# col_norms = np.linalg.norm(wt_fwd, axis=0)
# signal_trans = wt_fwd @ wt_fwd.T
signal_trans = fwd_field @ np.diag(fwd_sing**2) @ fwd_field.T

# signal_trans = fwd_full @ fwd_full.T
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
signal_cov = signal_trans @ data_cov['data'] @ signal_trans.T
noise_cov = noise_trans @ data_cov['data'] @ noise_trans.T

# Symmetrize (numerical hygiene)
signal_cov = (signal_cov + signal_cov.T) / 2
noise_cov = (noise_cov + noise_cov.T) / 2
denom = signal_cov + noise_cov
denom = (denom + denom.T) / 2

# Generalized eigendecomposition
eigenvalues, eigenvectors = eigh(signal_cov, denom)



# %% plot eigenvalues
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
plt.plot(np.sort(eigenvalues)[::-1], marker='o')
plt.title("GED Eigenspectrum")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
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

# %% select eiennvectors to form ZCA transform
threshold = .99
thresh_eigval = eigenvalues[eigenvalues >= threshold]
U_signal = eigenvectors[:, eigenvalues >= threshold]


# %% make projectors

desc_prefix = f"ZCA_extOrder{ext_order}_thresh{threshold:.2f}"

n_channels, n_signal = U_signal.shape
n_noise = n_channels - n_signal

if n_noise <= 0:
    raise ValueError(
        f"Signal subspace ({n_signal} dims) must be smaller than "
        f"channel space ({n_channels} dims) to create projectors."
    )

# Compute orthogonal complement (noise subspace)
# null_space(U_signal.T) gives vectors orthogonal to all rows of U_signal.T
U_noise = null_space(U_signal.T)

print(f"Signal subspace: {n_signal} dimensions")
print(f"Noise subspace (projectors): {U_noise.shape[1]} dimensions")

# Create projectors from noise subspace vectors
projs = []
for k in range(U_noise.shape[1]):
    vec = U_noise[:, k]
    vec /= np.linalg.norm(vec)  # Ensure unit norm
    
    proj_data = dict(
        col_names=list(sss_info['ch_names']),
        row_names=None,
        data=vec[np.newaxis, :],
        ncol=len(sss_info['ch_names']),
        nrow=1,
    )
    proj = mne.Projection(
        active=False,
        data=proj_data,
        desc=f"{desc_prefix}-{k+1:03d}",
    )
    projs.append(proj)

# %% add projectors to info and apply

data_proj = data.copy()
data_proj.add_proj(projs)


# %% compare to hfc

data_hfc = data.copy()
hfc_projs = mne.preprocessing.compute_proj_hfc(
    data_hfc.info,
    order=3,
    picks=mne.pick_types(data_hfc.info, meg=True, eeg=False, exclude='bads'),
)
data_hfc.add_proj(hfc_projs)

# %% plot before and after applying projectors


proj_psd = data_proj.compute_psd(fmax=100, n_jobs=-1)
projApplied_psd = data_proj.compute_psd(fmax=100, n_jobs=-1, proj=True)
hfcApplied_psd = data_hfc.compute_psd(fmax=100, n_jobs=-1, proj=True)


# plot with subplot
proj_psd.plot(show=True)
plt.title("no proj")

projApplied_psd.plot(show=True)
plt.title("ZCA proj")

hfcApplied_psd.plot(show=True)
plt.title("hfc proj")

plt.show()

# compare shielding between ZCA and HFC
# compute shielding factor as ratio of psd before and after proj
freqs = proj_psd.freqs
proj_shielding = proj_psd.get_data().mean(axis=0) / projApplied_psd.get_data().mean(axis=0)
hfc_shielding = proj_psd.get_data().mean(axis=0) / hfcApplied_psd.get_data().mean(axis=0)
plt.figure()
plt.semilogy(freqs+1, proj_shielding, label='ZCA proj', marker='o')
plt.semilogy(freqs, hfc_shielding, label='HFC proj', marker='o')
plt.title("Shielding Factor Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Shielding Factor (no proj / with proj)")
plt.legend()
plt.grid()
plt.show()


base_epochs = mne.Epochs(data, event_id="response/left", tmin=-.2, tmax=.2, baseline=None, preload=True)
base_epochs.plot_image()
plt.show()


# %%
