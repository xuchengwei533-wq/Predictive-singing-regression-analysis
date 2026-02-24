import numpy as np
import librosa
import os
import time
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

directory = r"D:\同济车辆\chest_audio\Chest new0206\Chest new0206"
Audio_path = directory

Jitter_out = os.path.join(directory, "JitterOutput")
os.makedirs(Jitter_out, exist_ok=True)

files = os.listdir(Audio_path)
wav_files = []
for f in files:
    if not f.lower().endswith(".wav"):
        continue
    in_path = os.path.join(Audio_path, f)
    if not os.path.isfile(in_path):
        continue
    wav_files.append(f)
wav_files.sort()

print(f"📁 输入目录: {Audio_path}")
print(f"📦 共检测到 WAV 文件: {len(wav_files)}")
print(f"📄 Jitter 输出目录: {Jitter_out}")
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
    jitter = diffs / denom
    return jitter

if HAS_TQDM:
    pbar = tq.tqdm(wav_files, total=len(wav_files), desc="提取Jitter", unit="file", dynamic_ncols=True)
    for idx, file in enumerate(pbar, start=1):
        start_t = time.perf_counter()
        in_path = os.path.join(Audio_path, file)
        out_name = os.path.splitext(file)[0] + ".csv"
        out_path = os.path.join(Jitter_out, out_name)
        if os.path.exists(out_path):
            pbar.set_postfix({"步骤": "跳过(已存在)", "文件": file})
            continue
        audio, original_sr, target_sr = load_audio(in_path)
        pbar.set_postfix({"步骤": "提取Jitter", "文件": file})
        jitter = extract_jitter(audio, target_sr)
        if jitter is None:
            tq.tqdm.write(f"❌ 提取失败: {file}")
            continue
        np.savetxt(out_path, jitter, delimiter=",", fmt="%.6f")
        cost_s = time.perf_counter() - start_t
        pbar.set_postfix({"步骤": f"完成({cost_s:.1f}s)", "文件": file})
        tq.tqdm.write(f"✅ [{idx}/{len(wav_files)}] {file} | Jitter saved: {out_path}")
else:
    for idx, file in enumerate(wav_files, start=1):
        start_t = time.perf_counter()
        in_path = os.path.join(Audio_path, file)
        out_name = os.path.splitext(file)[0] + ".csv"
        out_path = os.path.join(Jitter_out, out_name)
        if os.path.exists(out_path):
            print(f"[{idx}/{len(wav_files)}] 跳过(已存在): {file}")
            continue
        audio, original_sr, target_sr = load_audio(in_path)
        jitter = extract_jitter(audio, target_sr)
        if jitter is None:
            print(f"[{idx}/{len(wav_files)}] 失败: {file}")
            continue
        np.savetxt(out_path, jitter, delimiter=",", fmt="%.6f")
        cost_s = time.perf_counter() - start_t
        print(f"[{idx}/{len(wav_files)}] 完成({cost_s:.1f}s): {file}")

print("🎉 全部处理完成。")
