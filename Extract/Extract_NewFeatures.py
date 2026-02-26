import os
import time
import numpy as np
import librosa
try:
    import tqdm as tq
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False
try:
    import soundfile as sf
    HAS_SF = True
except Exception:
    HAS_SF = False
try:
    import parselmouth as pm
    HAS_PM = True
except Exception:
    HAS_PM = False

directory = r"D:\同济车辆\chest_audio\Chest new0206\Chest new0206"
audio_path = directory

outputs = {
    "Jitter": os.path.join(directory, "JitterOutput"),
    "Shimmer": os.path.join(directory, "ShimmerOutput"),
    "H1H2": os.path.join(directory, "H1H2Output"),
    "HNR": os.path.join(directory, "HNR_Output"),
    "QValue": os.path.join(directory, "QValueOutput"),
    "SpectralSlope": os.path.join(directory, "SpectralSlopeOutput"),
    "LowFreqEnergyRatio": os.path.join(directory, "LowFreqEnergyRatioOutput"),
    "HighFreqNoiseRatio": os.path.join(directory, "HighFreqNoiseRatioOutput"),
    "CPP": os.path.join(directory, "CPP_Output")
}

for path in outputs.values():
    os.makedirs(path, exist_ok=True)

files = os.listdir(audio_path)
wav_files = []
for f in files:
    if not f.lower().endswith(".wav"):
        continue
    in_path = os.path.join(audio_path, f)
    if not os.path.isfile(in_path):
        continue
    wav_files.append(f)
wav_files.sort()

print(f"📁 输入目录: {audio_path}")
print(f"📦 共检测到 WAV 文件: {len(wav_files)}")
for name, path in outputs.items():
    print(f"📄 {name} 输出目录: {path}")
if not HAS_PM:
    print("⚠️ 未检测到 parselmouth，HNR 使用近似计算方式")
print("-" * 60)

def load_audio(file_path, target_sr=44100):
    if HAS_SF:
        audio_raw, original_sr = sf.read(file_path, dtype="float32", always_2d=False)
        if isinstance(original_sr, np.ndarray):
            original_sr = int(original_sr)
        if hasattr(audio_raw, "ndim") and audio_raw.ndim > 1:
            audio_raw = np.mean(audio_raw, axis=1)
    else:
        audio_raw, original_sr = librosa.load(file_path, sr=None, mono=True)
    if original_sr != target_sr:
        audio = librosa.resample(audio_raw, orig_sr=original_sr, target_sr=target_sr)
    else:
        audio = audio_raw
    return audio, original_sr, target_sr

def extract_jitter(audio, sr, hop_length=512):
    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length
    )
    if f0 is None:
        return None
    f0 = np.asarray(f0, dtype=np.float32)
    mask = np.isfinite(f0) & (f0 > 0)
    f0_valid = f0[mask]
    if f0_valid.size < 2:
        return None
    periods = 1.0 / f0_valid
    diffs = np.abs(np.diff(periods))
    denom = np.mean(periods)
    if denom <= 0:
        return None
    return diffs / denom

def extract_shimmer(audio, sr, hop_length=512, frame_length=2048):
    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length
    )
    if f0 is None:
        return None
    f0 = np.asarray(f0, dtype=np.float32)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    if rms is None or rms.size == 0:
        return None
    min_len = min(f0.shape[0], rms.shape[0])
    f0 = f0[:min_len]
    rms = rms[:min_len]
    mask = np.isfinite(f0) & (f0 > 0)
    amp = rms[mask]
    if amp.size < 2:
        return None
    diffs = np.abs(np.diff(amp))
    denom = np.mean(amp)
    if denom <= 0:
        return None
    return diffs / denom

def extract_h1h2(audio, sr, hop_length=512, n_fft=2048):
    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length
    )
    if f0 is None:
        return None
    f0 = np.asarray(f0, dtype=np.float32)
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True))
    n_bins, n_frames = S.shape
    min_len = min(f0.shape[0], n_frames)
    f0 = f0[:min_len]
    S = S[:, :min_len]
    h1h2_vals = []
    for t in range(min_len):
        f0_t = f0[t]
        if not np.isfinite(f0_t) or f0_t <= 0:
            continue
        bin1 = int(np.round(f0_t * n_fft / sr))
        bin2 = int(np.round(2.0 * f0_t * n_fft / sr))
        if bin2 <= 0 or bin2 >= n_bins or bin1 <= 0 or bin1 >= n_bins:
            continue
        h1 = S[bin1, t]
        h2 = S[bin2, t]
        h1_db = 20.0 * np.log10(h1 + 1e-8)
        h2_db = 20.0 * np.log10(h2 + 1e-8)
        h1h2_vals.append(h1_db - h2_db)
    if len(h1h2_vals) == 0:
        return None
    return np.asarray(h1h2_vals, dtype=np.float32)

def extract_hnr(audio, sr, frame_length=2048, hop_length=512):
    if HAS_PM:
        snd = pm.Sound(audio, sampling_frequency=sr)
        harmonicity = snd.to_harmonicity(time_step=hop_length / float(sr))
        hnr = harmonicity.values[0]
        return hnr
    harmonic = librosa.effects.harmonic(audio)
    noise = audio - harmonic
    rms_h = librosa.feature.rms(y=harmonic, frame_length=frame_length, hop_length=hop_length)[0]
    rms_n = librosa.feature.rms(y=noise, frame_length=frame_length, hop_length=hop_length)[0]
    hnr = 10.0 * np.log10((rms_h ** 2) / (rms_n ** 2 + 1e-12) + 1e-12)
    return hnr

def extract_q_values(audio, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    q_vals = []
    for t in range(S.shape[1]):
        mag = S[:, t]
        if mag.size == 0:
            continue
        peak_idx = int(np.argmax(mag))
        peak_mag = mag[peak_idx]
        if not np.isfinite(peak_mag) or peak_mag <= 0:
            continue
        f0 = freqs[peak_idx]
        if not np.isfinite(f0) or f0 <= 0:
            continue
        target = peak_mag / np.sqrt(2.0)
        left = peak_idx
        while left > 0 and mag[left] >= target:
            left -= 1
        right = peak_idx
        while right < len(mag) - 1 and mag[right] >= target:
            right += 1
        if left == peak_idx or right == peak_idx:
            continue
        bw = freqs[right] - freqs[left]
        if not np.isfinite(bw) or bw <= 0:
            continue
        q_vals.append(float(f0 / bw))
    if len(q_vals) == 0:
        return None
    return np.asarray(q_vals, dtype=np.float32)

def extract_spectral_slope(audio, sr, hop_length=512, n_fft=2048):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True))
    if S is None or S.size == 0:
        return None
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    slopes = []
    for t in range(S.shape[1]):
        mag = S[:, t]
        log_mag = np.log10(mag + 1e-8)
        slope = np.polyfit(freqs, log_mag, 1)[0]
        slopes.append(slope)
    if len(slopes) == 0:
        return None
    return np.asarray(slopes, dtype=np.float32)

def extract_low_freq_energy_ratio(audio, sr, hop_length=512, n_fft=2048):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask = (freqs >= 0) & (freqs <= 500)
    total_mask = (freqs >= 0) & (freqs <= 1000)
    low_energy = np.sum(S[low_mask, :], axis=0)
    total_energy = np.sum(S[total_mask, :], axis=0)
    ratio = low_energy / (total_energy + 1e-12)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return None
    return ratio.astype(np.float32)

def extract_high_freq_noise_ratio(audio, sr, hop_length=512, n_fft=2048):
    harmonic = librosa.effects.harmonic(audio)
    noise = audio - harmonic
    S = np.abs(librosa.stft(noise, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    high_min = min(4000.0, 0.45 * sr)
    high_mask = freqs >= high_min
    high_energy = np.sum(S[high_mask, :], axis=0)
    total_energy = np.sum(S, axis=0)
    ratio = high_energy / (total_energy + 1e-12)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return None
    return ratio.astype(np.float32)

def extract_cpp(audio, sr, hop_length=512, n_fft=2048):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True))
    if S is None or S.size == 0:
        return None
    log_mag = np.log(S + 1e-8)
    cepstra = np.fft.irfft(log_mag, axis=0)
    quef = np.arange(cepstra.shape[0]) / float(sr)
    qmin = 1.0 / 400.0
    qmax = 1.0 / 60.0
    mask = (quef >= qmin) & (quef <= qmax)
    if not np.any(mask):
        return None
    cep_range = cepstra[mask, :]
    peak = np.max(cep_range, axis=0)
    baseline = np.mean(cep_range, axis=0)
    cpp = peak - baseline
    cpp = cpp[np.isfinite(cpp)]
    if cpp.size == 0:
        return None
    return cpp.astype(np.float32)

def save_series(out_dir, name, series):
    out_path = os.path.join(out_dir, name)
    np.savetxt(out_path, series, delimiter=",", fmt="%.6f")
    return out_path

def handle_one(file, audio, sr):
    base = os.path.splitext(file)[0] + ".csv"

    targets = [
        ("Jitter", outputs["Jitter"], extract_jitter, (audio, sr)),
        ("Shimmer", outputs["Shimmer"], extract_shimmer, (audio, sr)),
        ("H1H2", outputs["H1H2"], extract_h1h2, (audio, sr)),
        ("HNR", outputs["HNR"], extract_hnr, (audio, sr)),
        ("QValue", outputs["QValue"], extract_q_values, (audio, sr)),
        ("SpectralSlope", outputs["SpectralSlope"], extract_spectral_slope, (audio, sr)),
        ("LowFreqEnergyRatio", outputs["LowFreqEnergyRatio"], extract_low_freq_energy_ratio, (audio, sr)),
        ("HighFreqNoiseRatio", outputs["HighFreqNoiseRatio"], extract_high_freq_noise_ratio, (audio, sr)),
        ("CPP", outputs["CPP"], extract_cpp, (audio, sr))
    ]

    saved = []
    for name, out_dir, func, args in targets:
        out_path = os.path.join(out_dir, base)
        if os.path.exists(out_path):
            saved.append((name, "跳过"))
            continue
        series = func(*args)
        if series is None:
            saved.append((name, "失败"))
            continue
        save_series(out_dir, base, series)
        saved.append((name, "完成"))
    return saved

if HAS_TQDM:
    pbar = tq.tqdm(wav_files, total=len(wav_files), desc="提取特征", unit="file", dynamic_ncols=True)
    for idx, file in enumerate(pbar, start=1):
        start_t = time.perf_counter()
        in_path = os.path.join(audio_path, file)
        audio, original_sr, target_sr = load_audio(in_path)
        results = handle_one(file, audio, target_sr)
        cost_s = time.perf_counter() - start_t
        status = ",".join([f"{k}:{v}" for k, v in results])
        pbar.set_postfix({"步骤": f"{cost_s:.1f}s", "文件": file})
        tq.tqdm.write(f"✅ [{idx}/{len(wav_files)}] {file} | {status}")
else:
    for idx, file in enumerate(wav_files, start=1):
        start_t = time.perf_counter()
        in_path = os.path.join(audio_path, file)
        audio, original_sr, target_sr = load_audio(in_path)
        results = handle_one(file, audio, target_sr)
        cost_s = time.perf_counter() - start_t
        status = ",".join([f"{k}:{v}" for k, v in results])
        print(f"✅ [{idx}/{len(wav_files)}] {file} | {status} | {cost_s:.1f}s")

print("🎉 全部处理完成。")
