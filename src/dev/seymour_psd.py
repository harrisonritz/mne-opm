import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from osl_ephys.preprocessing.osl_wrappers import detect_artefacts  # type: ignore
from matplotlib.colors import LinearSegmentedColormap


def mark_bad_var(raw_downsampled):
    # Get the data array from the MNE object
    data = raw_downsampled.get_data()  # (channels, time)

    # Define chunk size (1-second chunks)
    chunk_size = int(raw_downsampled.info["sfreq"])  # 1 second chunk length
    num_chunks = data.shape[1] // chunk_size

    chunk_variances = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_data = data[:, start_idx:end_idx]
        chunk_var = np.var(chunk_data, axis=1).mean()  # Mean variance of the chunk
        chunk_variances.append(chunk_var)

    # Time points for the plot
    time_points = np.arange(
        0,
        (num_chunks) * (chunk_size / raw_downsampled.info["sfreq"]),
        chunk_size / raw_downsampled.info["sfreq"],
    )

    # Calculate the mean and standard deviation of the variance across all time points
    mean_var = np.median(chunk_variances)  # Median variance across channels
    std_var = np.std(chunk_variances)  # Standard deviation of variances across channels

    # Initial threshold for identifying large variance
    initial_threshold = mean_var + (2 * std_var)

    # Variable to store the final threshold
    final_threshold = {"value": initial_threshold}

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.3)  # Leave space for the slider and checkbox

    # Plot chunk variances and initial threshold
    ax.scatter(time_points, chunk_variances, color="blue", s=10, label="Variance")
    mean_line = ax.axhline(
        mean_var, color="green", linestyle="--", linewidth=1, label="Median Variance"
    )
    (threshold_line,) = ax.plot(
        [time_points[0], time_points[-1]],
        [initial_threshold, initial_threshold],
        "r--",
        lw=2,
        label="Threshold",
    )

    # Add labels and legend
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Variance")
    ax.set_title("Variance over Time with Adjustable Threshold")
    ax.legend(loc="upper left")

    # Slider setup
    ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
    threshold_slider = Slider(
        ax_slider,
        "Threshold",
        np.min(chunk_variances),
        np.max(chunk_variances),
        valinit=initial_threshold,
    )

    def update(val):
        # Update the threshold line when the slider value changes
        current_threshold = threshold_slider.val
        threshold_line.set_ydata([current_threshold, current_threshold])
        fig.canvas.draw_idle()

    # Link slider to update function
    threshold_slider.on_changed(update)

    # Checkbox setup for apply toggle
    ax_checkbox = plt.axes(
        [0.4, 0.05, 0.3, 0.1]
    )  # Larger checkbox, no yellow background, centered
    apply_checkbox = CheckButtons(ax_checkbox, ["Apply"], [False])
    apply_checkbox.ax.set_frame_on(False)  # Turn off the frame around the checkbuttons

    # Adjust font size for the checkbox labels
    for label in apply_checkbox.labels:
        label.set_fontsize(18)

    # Customize checkbox to remove border
    for artist in apply_checkbox.ax.get_children():
        if isinstance(artist, plt.Polygon):
            artist.set_edgecolor("none")

    def on_apply(label):
        # Toggle the final threshold action when the checkbox is clicked
        if apply_checkbox.get_status()[0]:
            # Store the final threshold value
            final_threshold["value"] = threshold_slider.val

            # Identify bad chunks based on the final threshold
            bad_chunks = []
            for i, chunk_var in enumerate(chunk_variances):
                if chunk_var > final_threshold["value"]:
                    onset = i * chunk_size / raw_downsampled.info["sfreq"]
                    duration = chunk_size / raw_downsampled.info["sfreq"]
                    bad_chunks.append((onset, duration))

            # Retrieve existing annotations
            existing_annotations = raw_downsampled.annotations

            # Find indices of annotations labeled as 'Bad Segment'
            bad_segment_indices = [
                idx
                for idx, anno in enumerate(existing_annotations)
                if anno["description"] == "Bad Segment"
            ]

            # If there are any 'Bad Segment' annotations, remove them
            if bad_segment_indices:
                raw_downsampled.annotations.delete(bad_segment_indices)

            # Calculate the percentage of time marked as bad
            total_time = raw_downsampled.times[-1] - raw_downsampled.times[0]
            bad_time = sum([bc[1] for bc in bad_chunks])
            bad_percentage = (bad_time / total_time) * 100

            # Print confirmation and percentage
            plt.close(fig)
            print(f"APPLIED: {bad_percentage:.2f}% of the time marked as bad")

            # Create new annotations
            new_annotations = mne.Annotations(
                onset=[bc[0] for bc in bad_chunks],
                duration=[bc[1] for bc in bad_chunks],
                description=["Bad Segment"] * len(bad_chunks),
            )

            # Combine existing annotations with new ones
            combined_onset = np.concatenate(
                [existing_annotations.onset, new_annotations.onset]
            )
            combined_duration = np.concatenate(
                [existing_annotations.duration, new_annotations.duration]
            )
            combined_description = np.concatenate(
                [existing_annotations.description, new_annotations.description]
            )

            # Set the combined annotations
            raw_downsampled.set_annotations(
                mne.Annotations(
                    onset=combined_onset,
                    duration=combined_duration,
                    description=combined_description,
                )
            )

            return raw_downsampled
        else:
            print("Apply action not enabled")
            return None

    # Link checkbox to the apply function
    apply_checkbox.on_clicked(on_apply)

    # Checkbox setup for log10 scale toggle
    ax_log_checkbox = plt.axes([0.01, 0.78, 0.1, 0.02])
    log_checkbox = CheckButtons(ax_log_checkbox, ["Log10\nScale"], [False])

    log_checkbox.ax.set_frame_on(False)  # Turn off the frame around the checkbuttons

    def toggle_log_scale(label):
        # Toggle the y-axis between log10 scale and linear
        if log_checkbox.get_status()[0]:  # Check if checkbox is selected
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")
        fig.canvas.draw_idle()

    # Link checkbox to the toggle function
    log_checkbox.on_clicked(toggle_log_scale)

    plt.show()


def detect_bad_chan_PSD(
    raw_downsampled, alpha=0.05, fmin=2, fmax=80, verbose=True, n_fft=2000
):
    """
    Compute Power Spectral Density (PSD) and optionally plot with a GESD Alpha Adjustment Slider.

    This function computes the PSD of the provided MNE Raw object within a specified frequency range.
    It allows interactive adjustment of the GESD alpha parameter to identify bad channels if `verbose` is True.
    If `verbose` is False, the function computes bad channels using the initial alpha value without plotting.

    Parameters
    ----------
    raw_downsampled : mne.io.Raw
        Downsampled MNE Raw object containing EEG/MEG data.
    initial_alpha : float
        Initial alpha value for the GESD outlier detection algorithm.
        Determines the threshold for identifying bad channels.
    fmin : float
        Minimum frequency (Hz) for PSD computation.
    fmax : float
        Maximum frequency (Hz) for PSD computation.
    verbose : bool, optional (default=True)
        If True, displays an interactive plot for GESD alpha adjustment.
        If False, skips plotting and directly computes bad channels.
    n_fft : int, optional (default=2000)
        Length of the FFT used for PSD computation. Higher values improve frequency resolution.

    Returns
    -------
    bad_chans_out : list of str
        List of channel names identified as bad channels based on the GESD algorithm.
    """

    # Compute PSD
    psd = raw_downsampled.compute_psd(
        fmin=fmin, fmax=fmax, n_fft=n_fft, reject_by_annotation=True, verbose=True
    )
    pow = psd.get_data()
    freq = psd.freqs

    # Get channel names and bad channels
    chan_names = np.array(raw_downsampled.ch_names)
    bad_channels = np.array(raw_downsampled.info["bads"])

    # Ensure compatibility when no bad channels exist
    if len(bad_channels) > 0:
        chan_names = np.array([x for x in chan_names if x not in bad_channels])

    # Initialize output for bad channels
    bad_chans_out = []

    def compute_bad_channels(alpha):
        """Detect bad channels using GESD with a given alpha value."""
        gesd_args = {"alpha": alpha}

        # Handle invalid values in the PSD data
        invalid_mask = np.any((pow == 0) | np.isnan(pow), axis=1)
        pow_clean = np.copy(pow)
        pow_clean[invalid_mask] = np.nan
        valid_mean = np.nanmean(pow_clean)
        valid_std = np.nanstd(pow_clean)
        outlier_value = valid_mean + 4 * valid_std
        pow_clean[invalid_mask] = outlier_value

        # Compute log10 of the cleaned data
        pow_log = np.log10(pow_clean)

        # Detect bad channels
        bad_channels_mask = detect_artefacts(
            pow_log, axis=0, reject_mode="dim", gesd_args=gesd_args
        )
        return chan_names[bad_channels_mask]

    # If verbose is False, directly compute and return bad channels
    if not verbose:
        bad_channels = compute_bad_channels(alpha)
        for g in bad_channels:
            bad_chans_out.append(g)
        return bad_chans_out

    # Create the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.3, right=0.8)

    # Plot all channels initially
    lines = [
        ax.semilogy(freq, data, alpha=0.1, label=chan, color="blue")[0]
        for data, chan in zip(pow, chan_names)
    ]

    # Plot formatting
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB)")
    ax.set_title("Log Power Spectral Density with GESD Alpha Adjustment")
    ax.set_xlim([fmin, fmax])
    ax.grid(True)

    # Add a slider for adjusting alpha
    ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
    alpha_slider = Slider(ax_slider, "GESD Alpha", 0, 0.5, valinit=alpha)

    # Checkbox setup for apply toggle
    ax_checkbox = plt.axes([0.4, 0.05, 0.3, 0.1])
    apply_checkbox = CheckButtons(ax_checkbox, ["Apply"], [False])
    apply_checkbox.ax.set_frame_on(False)

    # Adjust font size for the checkbox labels
    for label in apply_checkbox.labels:
        label.set_fontsize(18)

    def update_plot(val):
        """Update the plot based on the current alpha value."""
        alpha = alpha_slider.val
        bad_channels_current = compute_bad_channels(alpha)
        for i, line in enumerate(lines):
            if chan_names[i] in bad_channels_current:
                line.set_color("black")
                line.set_alpha(1.0)
            else:
                line.set_color("blue")
                line.set_alpha(0.1)
        fig.canvas.draw_idle()

    def on_apply(label):
        """Apply the current settings and close the plot."""
        nonlocal bad_chans_out
        if apply_checkbox.get_status()[0]:
            bad_channels = compute_bad_channels(alpha_slider.val)
            for g in bad_channels:
                bad_chans_out.append(g)
            plt.close(fig)
            return bad_chans_out

    # Attach callbacks
    alpha_slider.on_changed(update_plot)
    apply_checkbox.on_clicked(on_apply)

    # Show the plot
    plt.show()

    return bad_chans_out
