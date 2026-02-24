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

RMS_out = os.path.join(directory, "RMS_Output")
os.makedirs(RMS_out, exist_ok=True)

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
print(f"📄 RMS 输出目录: {RMS_out}")
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

def extract_rms(audio, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return rms

if HAS_TQDM:
    pbar = tq.tqdm(wav_files, total=len(wav_files), desc="提取RMS", unit="file", dynamic_ncols=True)
    for idx, file in enumerate(pbar, start=1):
        start_t = time.perf_counter()
        in_path = os.path.join(Audio_path, file)
        out_name = os.path.splitext(file)[0] + ".csv"
        out_path = os.path.join(RMS_out, out_name)
        if os.path.exists(out_path):
            pbar.set_postfix({"步骤": "跳过(已存在)", "文件": file})
            continue
        audio, original_sr, target_sr = load_audio(in_path)
        pbar.set_postfix({"步骤": "提取RMS", "文件": file})
        rms = extract_rms(audio)
        if rms is None:
            tq.tqdm.write(f"❌ 提取失败: {file}")
            continue
        np.savetxt(out_path, rms, delimiter=",", fmt="%.6f")
        cost_s = time.perf_counter() - start_t
        pbar.set_postfix({"步骤": f"完成({cost_s:.1f}s)", "文件": file})
        tq.tqdm.write(f"✅ [{idx}/{len(wav_files)}] {file} | RMS saved: {out_path}")
else:
    for idx, file in enumerate(wav_files, start=1):
        start_t = time.perf_counter()
        in_path = os.path.join(Audio_path, file)
        out_name = os.path.splitext(file)[0] + ".csv"
        out_path = os.path.join(RMS_out, out_name)
        if os.path.exists(out_path):
            print(f"[{idx}/{len(wav_files)}] 跳过(已存在): {file}")
            continue
        audio, original_sr, target_sr = load_audio(in_path)
        rms = extract_rms(audio)
        if rms is None:
            print(f"[{idx}/{len(wav_files)}] 失败: {file}")
            continue
        np.savetxt(out_path, rms, delimiter=",", fmt="%.6f")
        cost_s = time.perf_counter() - start_t
        print(f"[{idx}/{len(wav_files)}] 完成({cost_s:.1f}s): {file}")

print("🎉 全部处理完成。")
