import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
import noisereduce as nr


def clean_and_extend_audio(input_path, output_path, min_duration_sec=120):
    # Load audio
    y, sr = librosa.load(input_path, sr=None, mono=False)
    y = y.astype(np.float32)

    # High-pass filter
    def butter_highpass(cutoff_hz, sr, order=4):
        sos = butter(order, cutoff_hz / (0.5 * sr), btype="highpass", output="sos")
        return sos

    def apply_highpass(y, sr, cutoff_hz=80.0):
        if y.ndim == 1:
            return sosfilt(butter_highpass(cutoff_hz, sr), y)
        out = np.empty_like(y)
        for ch in range(y.shape[0]):
            out[ch] = sosfilt(butter_highpass(cutoff_hz, sr), y[ch])
        return out

    def find_noise_sample(y, sr, top_db=30):
        y_mono = y if y.ndim == 1 else np.mean(y, axis=0)
        non_silent = librosa.effects.split(y_mono, top_db=top_db)
        if non_silent.size > 0:
            silent = []
            last = 0
            for start, end in non_silent:
                if start > last:
                    silent.append((last, start))
                last = end
            if last < len(y_mono):
                silent.append((last, len(y_mono)))
            if silent:
                lengths = [e - s for s, e in silent]
                s, e = silent[int(np.argmax(lengths))]
                if e - s > int(0.1 * sr):
                    return y_mono[s:e]
        n = int(0.5 * sr)
        return y_mono[: min(n, len(y_mono))]

    def peak_normalize(y, peak_dbfs=-1.0, eps=1e-9):
        peak = np.max(np.abs(y)) + eps
        target = 10 ** (peak_dbfs / 20.0)
        return y * (target / peak)

    def reduce_noise_channel(y_ch, sr, noise_profile, prop_decrease=0.9):
        return nr.reduce_noise(
            y=y_ch,
            y_noise=noise_profile,
            sr=sr,
            stationary=False,
            prop_decrease=prop_decrease,
            time_mask_smooth_ms=50,
            freq_mask_smooth_hz=200
        )

    # Apply high-pass
    y_hp = apply_highpass(y, sr, cutoff_hz=80.0)
    # Build noise profile
    noise_profile = find_noise_sample(y_hp, sr, top_db=30)
    # Denoise
    if y_hp.ndim == 1:
        y_dn = reduce_noise_channel(y_hp, sr, noise_profile, prop_decrease=0.9)
    else:
        y_dn = np.vstack([
            reduce_noise_channel(y_hp[ch], sr, noise_profile, prop_decrease=0.9)
            for ch in range(y_hp.shape[0])
        ])
    # Normalize
    y_out = peak_normalize(y_dn, peak_dbfs=-1.0)
    # Extend audio if less than min_duration_sec
    duration = y_out.shape[-1] / sr
    if duration < min_duration_sec:
        reps = int(np.ceil(min_duration_sec / duration))
        y_out = np.tile(y_out, reps)
        # Truncate to exact min_duration_sec
        total_samples = int(min_duration_sec * sr)
        if y_out.ndim == 1:
            y_out = y_out[:total_samples]
        else:
            y_out = y_out[:, :total_samples]
    # Shape fix for writing
    data_to_write = y_out if y_out.ndim == 1 else y_out.T
    sf.write(output_path, data_to_write, sr, subtype="PCM_16")
    return output_path