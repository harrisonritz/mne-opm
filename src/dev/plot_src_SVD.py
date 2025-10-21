# %%
import mne
import mne_qt_browser
import numpy as np
import matplotlib.pyplot as plt

# %%
subject = 'sub-009_ses-01'
subjects_dir = '/Users/hr0283/freesurfer/TSXpilot'
epo_path = '/Users/hr0283/Brown Dropbox/Harrison Ritz/opm_data/data/TSXpilot/bids/derivatives/CSI__hfc-3_winRef/sub-009/ses-01/meg/sub-009_ses-01_task-TSXpilot_epo.fif'
fwd_path = '/Users/hr0283/Brown Dropbox/Harrison Ritz/opm_data/data/TSXpilot/bids/derivatives/CSI__hfc-3_winRef/sub-009/ses-01/meg/sub-009_ses-01_task-TSXpilot_fwd.fif'

epo = mne.read_epochs(epo_path, preload=True)
fwd = mne.read_forward_solution(fwd_path)

src = fwd['src']
leadfield = fwd['sol']['data']

print(f"Lead field shape: {leadfield.shape}")
# e.g., (192, 30000) for 192 channels and 10000 dipoles with free orientation

# Compute SVD
# L = U @ S @ Vt
# U: (n_channels, n_channels) - sensor space patterns
# S: (n_channels,) - singular values  
# Vt: (n_channels, 3*n_sources) - source space patterns (transposed)
U, s, Vt = np.linalg.svd(leadfield, full_matrices=False)

print(f"U shape: {U.shape}")
print(f"Singular values shape: {s.shape}")
print(f"Vt shape: {Vt.shape}")

# The right singular vectors are the rows of Vt
# Each row of Vt is a spatial pattern in source space
n_modes = 10

# ============================================================================
# STEP 4: Prepare for visualization
# ============================================================================
# Get source space vertex numbers
vertices = [src[0]['vertno'], src[1]['vertno']]  # left and right hemisphere
n_dipoles_lh = len(vertices[0])
n_dipoles_rh = len(vertices[1])
n_dipoles_total = n_dipoles_lh + n_dipoles_rh
print(f"Number of dipoles - LH: {n_dipoles_lh}, RH: {n_dipoles_rh}, Total: {n_dipoles_total}")

# For free orientation, we have 3 components per dipole
# We need to decide how to visualize: use vector magnitude or project to surface normal

def vector_magnitude(v_matrix, n_dipoles):
    """
    Compute magnitude of 3D vectors from flattened array.
    v_matrix: (n_modes, 3*n_dipoles)
    Returns: (n_modes, n_dipoles)
    """
    # Reshape to (n_modes, n_dipoles, 3)
    v_reshaped = v_matrix.reshape(v_matrix.shape[0], n_dipoles, 3)
    # Compute magnitude across the 3 components
    magnitudes = np.linalg.norm(v_reshaped, axis=2)
    return magnitudes

# Extract the first n_modes right singular vectors
V_modes = Vt[:n_modes, :]  # (n_modes, 3*n_dipoles)

# Compute magnitudes
mode_magnitudes = vector_magnitude(V_modes, n_dipoles_total)

# Split into left and right hemispheres
mode_magnitudes_lh = mode_magnitudes[:, :n_dipoles_lh]
mode_magnitudes_rh = mode_magnitudes[:, n_dipoles_lh:]

# ============================================================================
# STEP 5: Visualize each mode on the brain surface
# ============================================================================

def plot_singular_vector(mode_idx, mode_data_lh, mode_data_rh, 
                         vertices, subject, subjects_dir):
    """Plot a single source space singular vector on the brain."""
    
    # Create SourceEstimate object
    stc = mne.SourceEstimate(
        # concatenate left and right hemisphere 1-D arrays into a single column
        data=np.concatenate((mode_data_lh, mode_data_rh))[:, np.newaxis],  # (n_dipoles_total, 1)
        vertices=vertices,
        tmin=0,
        tstep=1,
        subject=subject
    )
    
    # Plot on inflated brain
    brain = stc.plot(
        subject=subject,
        subjects_dir=subjects_dir,
        hemi='both',
        surface='inflated',  # or 'pial', 'white'
        time_label=f'SVD Mode {mode_idx+1} (SV={s[mode_idx]:.2e})',
        clim=dict(kind='percent', lims=[90, 95, 99]),  # Adjust to highlight structure
        colormap='RdBu_r',
        background='white',
        foreground='black',
        smoothing_steps=10,
        time_viewer=False,
        title=f'Right Singular Vector #{mode_idx+1}',
        size=(800, 600)
    )
    
    return brain

# Plot all modes
brains = []
for i in range(n_modes):
    print(f"\nPlotting mode {i+1}/{n_modes}")
    print(f"  Singular value: {s[i]:.3e}")
    
    brain = plot_singular_vector(
        i,
        mode_magnitudes_lh[i],
        mode_magnitudes_rh[i],
        vertices,
        subject,
        subjects_dir
    )
    brains.append(brain)

    # Optional: save screenshot
    # brain.save_image(f'svd_mode_{i+1:02d}.png')

# ============================================================================
# STEP 6: Plot singular value spectrum
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(s, 'o-', linewidth=2, markersize=4)
sv_pct = 100 * np.cumsum(s) / np.sum(s)
ax.axvline(np.argmax(sv_pct >= 95) + 1, color='blue', linestyle='--', label=f'95% variance at mode {np.argmax(sv_pct >= 95) + 1}')
ax.axvline(np.argmax(sv_pct >= 99) + 1, color='darkblue', linestyle='--', label=f'99% variance at mode {np.argmax(sv_pct >= 99) + 1}')
ax.set_xlabel('Singular Vector Index', fontsize=12)
ax.set_ylabel('Singular Value', fontsize=12)
ax.set_title('Lead Field Singular Value Spectrum', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
# plt.savefig('singular_value_spectrum.png', dpi=300)
plt.show()



# ============================================================================
# ALTERNATIVE: Project to surface normal (for fixed orientation interpretation)
# ============================================================================
def project_to_surface_normal(v_matrix, fwd):
    """
    Project free-orientation vectors to surface normal direction.
    Useful for interpreting what patterns would look like with fixed orientation.
    """
    # Get source orientations (surface normals)
    source_nn = fwd['source_nn']  # (n_dipoles, 3)
    
    n_modes = v_matrix.shape[0]
    n_dipoles = source_nn.shape[0]
    
    # Reshape v_matrix to (n_modes, n_dipoles, 3)
    v_reshaped = v_matrix.reshape(n_modes, n_dipoles, 3)
    
    # Project onto surface normal for each dipole
    # projected[i, j] = v[i, j, :] · nn[j, :]
    projected = np.einsum('ijk,jk->ij', v_reshaped, source_nn)
    
    return projected

# Project modes to surface normal
V_projected = project_to_surface_normal(V_modes, fwd)

# Split and plot
V_proj_lh = V_projected[:, :n_dipoles_lh]
V_proj_rh = V_projected[:, n_dipoles_lh:]

# Plot surface-normal-projected version
for i in range(min(3, n_modes)):  # Just plot first 3 as example
    stc = mne.SourceEstimate(
        # concatenate projected left/right hemisphere vectors into a single column
        data=np.concatenate((V_proj_lh[i], V_proj_rh[i]))[:, np.newaxis],
        vertices=vertices,
        tmin=0,
        tstep=1,
        subject=subject
    )
    
    brain = stc.plot(
        subject=subject,
        subjects_dir=subjects_dir,
        hemi='both',
        surface='inflated',
        time_label=f'Mode {i+1} (Normal Projection)',
        clim=dict(kind='value', pos_lims=[0, 50, 99]),
        colormap='RdBu_r',
        background='white',
        smoothing_steps=10
    )
    # brain.save_image(f'svd_mode_{i+1:02d}_normal_proj.png')

print("\nDone! All modes plotted.")


# %%
