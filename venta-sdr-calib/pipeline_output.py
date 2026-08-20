"""Shared log parsing, calibration, plots, and science-product output."""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import stats


def load_observation_log(log_path, output_dir):
    """Parse one SDR observation log and retain the readable parsed form."""
    from ExperimentsLogReader.experimentsLogReader import LogReaderFactory, LogTypes

    prettylogs = Path(output_dir) / "prettylogs"
    prettylogs.mkdir(parents=True, exist_ok=True)
    return LogReaderFactory.getLogReader(
        LogTypes.SDR,
        str(log_path),
        str(prettylogs),
    ).getLogs()


def observation_parameters(logs):
    """Return the FFT size and pre-LO frequency axis defined by the log."""
    fft_size = int(logs["header"]["Fs,Ns,RBW"][1])
    bandwidth = float(logs["header"]["Fs,Ns,RBW"][0])
    resolution = float(logs["header"]["Fs,Ns,RBW"][2])
    intermediate_frequency = float(logs["header"]["f_obs,LO,IF"][2])
    start = intermediate_frequency - bandwidth / 2.0
    frequency = start + np.arange(fft_size, dtype=np.float64) * resolution
    return fft_size, frequency


def wavelet_transform(data, wavelet, threshold):
    """Reproduce the established Tsys smoothing operation."""
    descriptor = pywt.Wavelet(wavelet)
    maximum_level = pywt.dwt_max_level(len(data), descriptor.dec_len)
    coefficients = pywt.wavedec(data, wavelet, level=maximum_level)
    for index in range(1, len(coefficients)):
        coefficients[index] = pywt.threshold(
            coefficients[index], threshold * max(coefficients[index])
        )
    return pywt.waverec(coefficients, wavelet)


def frequency_shifting(
    p_sig_left,
    p_sig_right,
    p_ref_left,
    p_ref_right,
    p_sig_on_left,
    p_sig_on_right,
    p_ref_on_left,
    p_ref_on_right,
    frequency,
    scan,
    logs,
    predicted_tsys=None,
):
    """Apply the established unbiased frequency-switching calibration."""
    df_div = float(logs["header"]["df_div,df"][0])
    bandwidth = float(logs["header"]["Fs,Ns,RBW"][0])
    frequency_shift = bandwidth / df_div
    spectrum_length = len(frequency)
    frequency_step = (
        frequency[spectrum_length - 1] - frequency[0]
    ) / (spectrum_length - 1)
    shift_bins = int(np.rint(frequency_shift / frequency_step))
    start_index = int(spectrum_length / 2 - spectrum_length * 0.5 / 2)
    end_index = int(spectrum_length / 2 + spectrum_length * 0.5 / 2)

    if predicted_tsys is None:
        tsys_off_1_left = float(logs["header"]["Tcal"][0]) * (
            (p_ref_on_left + p_ref_left)
            - np.mean(p_ref_on_left[start_index:end_index] - p_ref_left[start_index:end_index])
        ) / (
            2
            * np.mean(
                p_ref_on_left[start_index:end_index]
                - p_ref_left[start_index:end_index]
            )
        )
        tsys_off_2_left = float(logs["header"]["Tcal"][1]) * (
            (p_sig_on_left + p_sig_left)
            - np.mean(p_sig_on_left[start_index:end_index] - p_sig_left[start_index:end_index])
        ) / (
            2
            * np.mean(
                p_sig_on_left[start_index:end_index]
                - p_sig_left[start_index:end_index]
            )
        )
        tsys_off_1_right = float(logs["header"]["Tcal"][0]) * (
            (p_ref_on_right + p_ref_right)
            - np.mean(
                p_ref_on_right[start_index:end_index]
                - p_ref_right[start_index:end_index]
            )
        ) / (
            2
            * np.mean(
                p_ref_on_right[start_index:end_index]
                - p_ref_right[start_index:end_index]
            )
        )
        tsys_off_2_right = float(logs["header"]["Tcal"][1]) * (
            (p_sig_on_right + p_sig_right)
            - np.mean(
                p_sig_on_right[start_index:end_index]
                - p_sig_right[start_index:end_index]
            )
        ) / (
            2
            * np.mean(
                p_sig_on_right[start_index:end_index]
                - p_sig_right[start_index:end_index]
            )
        )
    else:
        print("Scan %d is using predicted system temperatures" % scan, flush=True)
        # Scan identifiers in the observation log are one-based; arrays are not.
        prediction_index = scan - 1
        tsys_off_1_left = np.array(predicted_tsys[0][prediction_index]) / 2
        tsys_off_2_left = np.array(predicted_tsys[1][prediction_index]) / 2
        tsys_off_1_right = np.array(predicted_tsys[2][prediction_index]) / 2
        tsys_off_2_right = np.array(predicted_tsys[3][prediction_index]) / 2

    ta_1_caloff_left = tsys_off_1_left * (p_sig_left - p_ref_left) / p_ref_left
    ta_1_caloff_right = tsys_off_1_right * (p_sig_right - p_ref_right) / p_ref_right
    ta_2_caloff_left = tsys_off_2_left * (p_ref_left - p_sig_left) / p_sig_left
    ta_2_caloff_right = tsys_off_2_right * (p_ref_right - p_sig_right) / p_sig_right

    ta_1_calon_left = (
        tsys_off_1_left + float(logs["header"]["Tcal"][0])
    ) * (p_sig_on_left - p_ref_on_left) / p_ref_on_left
    ta_1_calon_right = (
        tsys_off_1_right + float(logs["header"]["Tcal"][1])
    ) * (p_sig_on_right - p_ref_on_right) / p_ref_on_right
    ta_2_calon_left = (
        tsys_off_2_left + float(logs["header"]["Tcal"][0])
    ) * (p_ref_on_left - p_sig_on_left) / p_sig_on_left
    ta_2_calon_right = (
        tsys_off_2_right + float(logs["header"]["Tcal"][1])
    ) * (p_ref_on_right - p_sig_on_right) / p_sig_on_right

    ta_sig_left = (ta_1_caloff_left + ta_1_calon_left) / 2
    ta_sig_right = (ta_1_caloff_right + ta_1_calon_right) / 2
    ta_ref_left = (ta_2_caloff_left + ta_2_calon_left) / 2
    ta_ref_right = (ta_2_caloff_right + ta_2_calon_right) / 2

    ta_sig_left = np.roll(ta_sig_left, +shift_bins)
    ta_sig_right = np.roll(ta_sig_right, +shift_bins)
    ta_ref_left = np.roll(ta_ref_left, -shift_bins)
    ta_ref_right = np.roll(ta_ref_right, -shift_bins)
    ta_left = (ta_sig_left + ta_ref_left) / 2
    ta_right = (ta_sig_right + ta_ref_right) / 2

    keys = (
        (str(scan) + "s1", str(scan) + "r1"),
        (str(scan) + "r0", str(scan) + "s0"),
    )
    elevation = (
        float(logs[keys[0][0]]["AzEl"][1])
        + float(logs[keys[0][1]]["AzEl"][1])
        + float(logs[keys[1][0]]["AzEl"][1])
        + float(logs[keys[1][1]]["AzEl"][1])
    ) / 4
    elevation_polynomial = [
        float(value) for value in logs["header"]["Elev_poly"]
    ][::-1]
    sf_left = ta_left / (
        float(logs["header"]["DPFU"][0])
        * np.polyval(elevation_polynomial, elevation)
    )
    sf_right = ta_right / (
        float(logs["header"]["DPFU"][1])
        * np.polyval(elevation_polynomial, elevation)
    )
    return (
        sf_left[start_index:end_index],
        sf_right[start_index:end_index],
        frequency[start_index:end_index],
    )


def calibrate_and_write_outputs(
    *,
    logs,
    output_dir,
    frequency,
    spectra,
    files_total,
    start_time,
    framework,
    configuration=None,
):
    """Calibrate accumulated spectra and write the established science products."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiment = logs["header"]["exp_name"]
    s0, s1, r0, r1 = (
        spectra["s0"],
        spectra["s1"],
        spectra["r0"],
        spectra["r1"],
    )
    if not (len(s0) == len(s1) == len(r0) == len(r1)) or len(s0) % 2:
        raise ValueError("calibration state/channel lists are not complete")

    spectrum_length = len(frequency)
    start_index = int(spectrum_length / 2 - spectrum_length * 0.5 / 2)
    end_index = int(spectrum_length / 2 + spectrum_length * 0.5 / 2)
    local_oscillator = float(logs["header"]["f_obs,LO,IF"][1])
    frequency_with_lo = np.asarray(frequency) + local_oscillator

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    figure.suptitle(experiment + " raw data plot")
    for index in range(len(s0)):
        axis = axes[index % 2]
        for state in (s0, r0, s1, r1):
            axis.plot(frequency_with_lo, state[index]["data"])
    for axis, title in zip(axes, ("LCP", "RCP")):
        axis.set_xlabel("Frequency (MHz) with LO")
        axis.set_ylabel("Amplitude")
        axis.set_title(title)
        axis.grid(True)
    raw_plot = output / (experiment + "_raw_plots.png")
    figure.tight_layout()
    figure.savefig(raw_plot, dpi=150)
    plt.close(figure)

    tcal = [float(logs["header"]["Tcal"][0]), float(logs["header"]["Tcal"][1])]
    tsys = [[], [], [], []]
    for index in range(0, len(s0), 2):
        values = (
            tcal[0]
            * ((r1[index]["data"] + r0[index]["data"])
               - np.mean(r1[index]["data"][start_index:end_index]
                         - r0[index]["data"][start_index:end_index]))
            / (2 * np.mean(r1[index]["data"][start_index:end_index]
                           - r0[index]["data"][start_index:end_index])),
            tcal[1]
            * ((s1[index]["data"] + s0[index]["data"])
               - np.mean(s1[index]["data"][start_index:end_index]
                         - s0[index]["data"][start_index:end_index]))
            / (2 * np.mean(s1[index]["data"][start_index:end_index]
                           - s0[index]["data"][start_index:end_index])),
            tcal[0]
            * ((r1[index + 1]["data"] + r0[index + 1]["data"])
               - np.mean(r1[index + 1]["data"][start_index:end_index]
                         - r0[index + 1]["data"][start_index:end_index]))
            / (2 * np.mean(r1[index + 1]["data"][start_index:end_index]
                           - r0[index + 1]["data"][start_index:end_index])),
            tcal[1]
            * ((s1[index + 1]["data"] + s0[index + 1]["data"])
               - np.mean(s1[index + 1]["data"][start_index:end_index]
                         - s0[index + 1]["data"][start_index:end_index]))
            / (2 * np.mean(s1[index + 1]["data"][start_index:end_index]
                           - s0[index + 1]["data"][start_index:end_index])),
        )
        for polarization, value in enumerate(values):
            tsys[polarization].append(np.mean(value[start_index:end_index]))

    figure, axis = plt.subplots(figsize=(10, 5))
    for values in tsys:
        axis.plot(values)
    axis.set_xlabel("Scan")
    axis.set_ylabel("System temperature")
    axis.grid(True)
    tsys_plot = output / (experiment + "_tsys_plots.png")
    figure.tight_layout()
    figure.savefig(tsys_plot, dpi=150)
    plt.close(figure)

    predicted = [wavelet_transform(values, "rbio1.5", 3) for values in tsys]
    outliers = [np.where(np.abs(stats.zscore(np.asarray(values))) > 2)[0] for values in tsys]
    outlier_count = sum(len(values) for values in outliers)
    print("Tsys outliers found: %d" % outlier_count, flush=True)

    calibrated_left = []
    calibrated_right = []
    damaged_scans = []
    calibrated_frequency = None
    for index in range(0, len(s0), 2):
        scan = index // 2 + 1
        # scipy.stats reports zero-based positions, unlike observation scan IDs.
        scan_index = scan - 1
        is_good = all(scan_index not in values for values in outliers)
        correction = None if is_good else predicted
        left, right, calibrated_frequency = frequency_shifting(
            s0[index]["data"], s0[index + 1]["data"],
            r0[index]["data"], r0[index + 1]["data"],
            s1[index]["data"], s1[index + 1]["data"],
            r1[index]["data"], r1[index + 1]["data"],
            frequency, scan, logs, correction,
        )
        calibrated_left.append(left)
        calibrated_right.append(right)
        if not is_good:
            damaged_scans.append({
                "scan": scan,
                "timestamp": logs.get(str(scan) + "s0", {}).get("date", ""),
            })

    left_by_scan = np.asarray(calibrated_left, dtype=np.float64)
    right_by_scan = np.asarray(calibrated_right, dtype=np.float64)
    np.save(output / (experiment + "_left.npy"), left_by_scan)
    np.save(output / (experiment + "_right.npy"), right_by_scan)

    means_left = left_by_scan.mean(axis=1)
    means_right = right_by_scan.mean(axis=1)
    good_mask = (np.abs(means_left) < 2.0) & (np.abs(means_right) < 2.0)
    filtered = int((~good_mask).sum())
    if good_mask.any():
        left_average = left_by_scan[good_mask].mean(axis=0)
        right_average = right_by_scan[good_mask].mean(axis=0)
    else:
        print("[WARN] All scans failed the mean-flux filter; averaging all scans", flush=True)
        left_average = left_by_scan.mean(axis=0)
        right_average = right_by_scan.mean(axis=0)
        filtered = left_by_scan.shape[0]

    calibrated_frequency = np.asarray(calibrated_frequency) + local_oscillator
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    figure.suptitle(
        "%s — Calibrated spectra (mean of %d scans; filtered %d)"
        % (experiment, left_by_scan.shape[0] - filtered, filtered)
    )
    axes[0].plot(calibrated_frequency, left_average)
    axes[0].set_title("LCP")
    axes[0].set_ylabel("Flux density (Jy)")
    axes[1].plot(calibrated_frequency, right_average)
    axes[1].set_title("RCP")
    for axis in axes:
        axis.set_xlabel("Frequency (MHz)")
        axis.grid(True)
    split_plot = output / (experiment + "_dat.png")
    figure.tight_layout()
    figure.savefig(split_plot, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(calibrated_frequency, left_average, label="LCP")
    axis.plot(calibrated_frequency, right_average, label="RCP")
    for marker in (1665.402, 1667.359):
        axis.axvline(x=marker, linestyle=":", linewidth=1)
    axis.set_xlabel("Frequency (MHz)")
    axis.set_ylabel("Spectral flux density (Jy)")
    axis.set_title(experiment + " — Calibrated spectra")
    axis.grid(True)
    axis.legend()
    combined_plot = output / (experiment + "_dat_combined.png")
    figure.tight_layout()
    figure.savefig(combined_plot, dpi=150)
    plt.close(figure)

    product = {
        "left": left_average.tolist(),
        "right": right_average.tolist(),
        "xaxis": calibrated_frequency.tolist(),
    }
    json_path = output / (experiment + "_raw.json")
    json_path.write_text(json.dumps(product), encoding="utf-8")
    dat_path = output / (experiment + "_raw.dat")
    with dat_path.open("w", encoding="utf-8") as stream:
        for x_value, left, right in zip(
            calibrated_frequency, left_average, right_average
        ):
            stream.write("%.6f %.9g %.9g\n" % (x_value, left, right))

    dated_entries = [
        value.get("date", "")
        for key, value in logs.items()
        if key != "header" and isinstance(value, dict) and value.get("date")
    ]
    warnings = []
    if outlier_count:
        warnings.append("Damaged scans detected")
    expected_files = int(logs["header"]["N_scans,N_cal"][0]) * 8
    if files_total != expected_files:
        warnings.append("File count does not match observation log")
    run_log = {
        "experiment": experiment,
        "framework": framework,
        "configuration": configuration or {},
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", start_time),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "file_count": files_total,
        "scan_count": len(left_by_scan),
        "experiment_start": dated_entries[0] if dated_entries else "",
        "experiment_end": dated_entries[-1] if dated_entries else "",
        "tsys_anomalies": outlier_count,
        "filtered_scans": filtered,
        "warnings": warnings,
        "damaged_scans": damaged_scans,
    }
    log_path = output / (experiment + "_calibr_log.json")
    log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    paths = (
        raw_plot, tsys_plot, split_plot, combined_plot, json_path, dat_path, log_path
    )
    for path in paths:
        print("[OUTPUT] %s" % path, flush=True)
    return paths
