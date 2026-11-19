"""生成模具辨认演示用 WAV（16kHz/16bit/单声道），供手工冒烟测试。

用法：python scripts/gen_mold_demo.py <输出目录>
生成 2 种节拍完全相同、仅冲击频谱不同的模具（各 3 段已确认正常样音），
以及 3 段待辨录音：正常音量模具 A、低音量+起始静音的模具 B、
不属于任何模具的异频录音。
"""
from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np

SR = 16_000
# 两种模具节拍完全一样（换模后节拍相同的现场场景）
BEAT = [0.6, 1.2, 1.8, 2.4]
MOLD_SPECTRA = {
    "A": ([800.0, 1600.0], [1.0, 0.7]),
    "B": ([2600.0, 4200.0], [1.0, 0.8]),
}


def synth(
    times, freqs, weights, duration, gain=0.8, leading_silence=0.0,
    noise=0.002, decay=60.0, seed=11, burst_ms=55,
):
    rng = np.random.default_rng(seed)
    n = int((duration + leading_silence) * SR)
    x = rng.standard_normal(n).astype(np.float32) * noise
    burst_n = int(burst_ms / 1000 * SR)
    tt = np.arange(burst_n) / SR
    shape = np.zeros(burst_n, dtype=np.float32)
    for f, w in zip(freqs, weights):
        shape += w * np.sin(2 * np.pi * f * tt).astype(np.float32)
    shape *= np.exp(-decay * tt).astype(np.float32)
    peak = float(np.max(np.abs(shape)))
    if peak > 0:
        shape /= peak

    for t in times:
        s = int(round((t + leading_silence) * SR))
        e = min(n, s + burst_n)
        if s < n:
            x[s:e] += gain * shape[: e - s]
    return np.clip(x, -1, 1)


def save(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((samples * 32767).astype("<i2").tobytes())


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for mold, (freqs, weights) in MOLD_SPECTRA.items():
        for k in range(3):
            w = [x * (0.9 + 0.05 * k) for x in weights]
            save(
                out / f"mold_{mold}_{k + 1}.wav",
                synth(BEAT, freqs, w, 3.0, gain=0.5 + 0.1 * k, seed=50 + k),
            )
    # 待辨 1：模具 A，正常录音
    save(out / "unknown_A.wav", synth(BEAT, *MOLD_SPECTRA["A"], 3.0, seed=4243))
    # 待辨 2：模具 B，但整体调低音量且加 0.7s 起始静音——归属应仍为 B
    save(
        out / "unknown_B_quiet_silence.wav",
        synth(
            BEAT, *MOLD_SPECTRA["B"], 3.0,
            gain=0.1, leading_silence=0.7, seed=4242,
        ),
    )
    # 待辨 3：异频冲击，不属于任何已建模模具，应判超范围失败
    save(
        out / "unknown_foreign.wav",
        synth(BEAT, [5500.0], [1.0], 3.0, seed=77),
    )
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
