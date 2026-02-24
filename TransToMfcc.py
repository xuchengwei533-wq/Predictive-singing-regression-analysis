import numpy as np
import librosa
import os
import time
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display as ldisplay
    HAS_MPL = True
except Exception:
    HAS_MPL = False
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False
try:
    import tqdm as tq
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# 可选：如果你想把重采样后的音频也保存出来
SAVE_RESAMPLED_WAV = False

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    SAVE_RESAMPLED_WAV = False  # 没装 soundfile 就自动不保存重采样音频


def extract_mfcc_features(file_path, max_pad_len=128, target_sr=44100, n_mfcc=128, hop_length=512):
    """
    读取音频 -> 统一重采样到 target_sr -> 提取 MFCC
    返回:
        mfccs, original_sr, target_sr
    """
    try:
        # 先用原采样率读一次，获取 original_sr（用于输出）
        if HAS_SF:
            audio_raw, original_sr = sf.read(file_path, dtype="float32", always_2d=False)
            if isinstance(original_sr, np.ndarray):
                original_sr = int(original_sr)
            if hasattr(audio_raw, "ndim") and audio_raw.ndim > 1:
                audio_raw = np.mean(audio_raw, axis=1)
        else:
            audio_raw, original_sr = librosa.load(file_path, sr=None, mono=True)

        # 再重采样到 44100
        if original_sr != target_sr:
            audio = librosa.resample(audio_raw, orig_sr=original_sr, target_sr=target_sr)
        else:
            audio = audio_raw

        # 计算 MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=n_mfcc, hop_length=hop_length)

        # pad / truncate 到固定帧数
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.astype(np.float32), original_sr, target_sr, audio

    except Exception as e:
        print(f"❌ Error while parsing: {file_path}\nException: {e}")
        return None, None, None, None


directory = r"D:\同济车辆\chest_audio\Chest new0206\Chest new0206"
Audio_path = directory

MFCC_img_out = os.path.join(directory, "MFCC_Images")
os.makedirs(MFCC_img_out, exist_ok=True)

Resampled_out = os.path.join(directory, "Audio_44100")
if SAVE_RESAMPLED_WAV:
    os.makedirs(Resampled_out, exist_ok=True)

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
print(f"🖼️ MFCC 图像输出目录: {MFCC_img_out}")
if not HAS_SF:
    print("⚠️ 未检测到 soundfile，部分 WAV 读取可能很慢或在坏文件上卡住。建议安装: pip install soundfile")
if SAVE_RESAMPLED_WAV:
    print(f"🎧 重采样音频输出目录: {Resampled_out} (44100Hz)")
print("-" * 60)

if HAS_TQDM:
    pbar = tq.tqdm(wav_files, total=len(wav_files), desc="处理音频文件", unit="file", dynamic_ncols=True)
    for idx, file in enumerate(pbar, start=1):
        in_path = os.path.join(Audio_path, file)
        start_t = time.perf_counter()
        tq.tqdm.write(f"[{idx}/{len(wav_files)}] 开始: {file}")
        pbar.set_postfix({"步骤": "加载音频", "文件": file})
        mfccs, original_sr, target_sr, audio_44100 = extract_mfcc_features(
            in_path,
            max_pad_len=128,
            target_sr=44100,
            n_mfcc=128
        )
        if mfccs is None:
            continue
        out_name = os.path.splitext(file)[0] + ".png"
        out_path = os.path.join(MFCC_img_out, out_name)
        if os.path.exists(out_path):
            pbar.set_postfix({"步骤": "跳过(已存在)", "文件": file})
            continue
        pbar.set_postfix({"步骤": "保存图像", "文件": file})
        if HAS_MPL:
            plt.figure(figsize=(6, 4), dpi=150, facecolor='black')
            mfccs_show = np.maximum(mfccs, 0.0)
            ldisplay.specshow(mfccs_show, x_axis='time', sr=target_sr, hop_length=512, cmap='afmhot', vmin=0)
            plt.axis('off')
            plt.gca().set_facecolor('black')
            plt.tight_layout(pad=0)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        elif HAS_PIL:
            x = mfccs
            x_min = float(x.min())
            x_max = float(x.max())
            img = (255.0 * (x - x_min) / (x_max - x_min + 1e-8)).astype(np.uint8)
            Image.fromarray(img).save(out_path)
        else:
            tq.tqdm.write(f"❌ 无法保存图像: {file}")
            continue
        if SAVE_RESAMPLED_WAV and HAS_SF:
            pbar.set_postfix({"步骤": "保存WAV", "文件": file})
            wav_out_path = os.path.join(Resampled_out, file)
            sf.write(wav_out_path, audio_44100, target_sr, subtype="PCM_16")
        cost_s = time.perf_counter() - start_t
        pbar.set_postfix({"步骤": f"完成({cost_s:.1f}s)", "文件": file})
        tq.tqdm.write(
            f"✅ [{idx}/{len(wav_files)}] {file} | 原采样率 {original_sr} Hz -> {target_sr} Hz | 图像 {out_path}"
            + (f" | WAV {wav_out_path}" if SAVE_RESAMPLED_WAV and HAS_SF else "")
        )
else:
    for idx, file in enumerate(wav_files, start=1):
        in_path = os.path.join(Audio_path, file)
        start_t = time.perf_counter()
        print(f"[{idx}/{len(wav_files)}] 加载音频: {file}", flush=True)
        mfccs, original_sr, target_sr, audio_44100 = extract_mfcc_features(
            in_path,
            max_pad_len=128,
            target_sr=44100,
            n_mfcc=128
        )
        if mfccs is None:
            print(f"[{idx}/{len(wav_files)}] 失败: {file}")
            continue
        out_name = os.path.splitext(file)[0] + ".png"
        out_path = os.path.join(MFCC_img_out, out_name)
        if os.path.exists(out_path):
            print(f"[{idx}/{len(wav_files)}] 跳过(已存在): {out_name}")
            continue
        print(f"[{idx}/{len(wav_files)}] 保存图像: {out_name}", flush=True)
        if HAS_MPL:
            plt.figure(figsize=(6, 4), dpi=150, facecolor='black')
            mfccs_show = np.maximum(mfccs, 0.0)
            ldisplay.specshow(mfccs_show, x_axis='time', sr=target_sr, hop_length=512, cmap='afmhot', vmin=0)
            plt.axis('off')
            plt.gca().set_facecolor('black')
            plt.tight_layout(pad=0)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        elif HAS_PIL:
            x = mfccs
            x_min = float(x.min())
            x_max = float(x.max())
            img = (255.0 * (x - x_min) / (x_max - x_min + 1e-8)).astype(np.uint8)
            Image.fromarray(img).save(out_path)
        else:
            print(f"❌ 无法保存图像: {file}")
            continue
        if SAVE_RESAMPLED_WAV and HAS_SF:
            print(f"[{idx}/{len(wav_files)}] 保存WAV: {file}", flush=True)
            wav_out_path = os.path.join(Resampled_out, file)
            sf.write(wav_out_path, audio_44100, target_sr, subtype="PCM_16")
        cost_s = time.perf_counter() - start_t
        print(f"[{idx}/{len(wav_files)}] 完成({cost_s:.1f}s): {file}")

print("🎉 全部处理完成。")
