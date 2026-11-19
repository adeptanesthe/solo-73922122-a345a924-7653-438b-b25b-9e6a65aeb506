"""合成测试 WAV 与 TestClient 夹具。"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.service import Store

SR = 16_000


def build_wav(
    impact_times: list[float],
    duration: float | None = None,
    sr: int = SR,
    gain: float = 0.8,
    burst_ms: int = 12,
    echoes: dict[float, float] | None = None,
    echo_gain: float = 0.7,
    noise_floor: float = 0.0,
    sample_width: int = 2,
    channels: int = 1,
    seed: int = 7,
) -> bytes:
    """生成带若干冲击噪声脉冲的 PCM WAV。

    echoes: {主冲击时刻: 回声延迟秒}，回声幅度约为主冲击的 40%。
    """
    rng = np.random.default_rng(seed)
    if duration is None:
        duration = max(impact_times) + 0.4 if impact_times else 1.0
    n = int(round(duration * sr))
    x = rng.standard_normal(n).astype(np.float32) * noise_floor
    burst_n = int(burst_ms / 1000 * sr)
    decay = np.exp(-np.linspace(0, 5, burst_n)).astype(np.float32)

    def put(t: float, amp: float) -> None:
        start = int(round(t * sr))
        end = min(n, start + burst_n)
        if start >= n or end <= 0:
            return
        seg = rng.standard_normal(end - start).astype(np.float32)
        x[start:end] += amp * seg * decay[: end - start]

    for t in impact_times:
        put(t, gain)
    for t, delay in (echoes or {}).items():
        put(t + delay, gain * echo_gain)

    x = np.clip(x, -1.0, 1.0)

    if sample_width == 2:
        pcm = (x * 32767).astype("<i2").tobytes()
        sampwidth = 2
    else:  # 8-bit unsigned PCM
        pcm = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8).tobytes()
        sampwidth = 1
    if channels == 2:
        pcm = b"".join(bytes([b, b]) for b in pcm) if sampwidth == 1 else b""
        if sampwidth == 2:
            stereo = np.repeat((x * 32767).astype("<i2"), 2)
            pcm = stereo.tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.store", Store(tmp_path))
    return TestClient(app)


def post_inspect(client: TestClient, ref: bytes, test: bytes):
    return client.post(
        "/api/inspect",
        files={
            "reference": ("ref.wav", ref, "audio/wav"),
            "test": ("test.wav", test, "audio/wav"),
        },
    )


# 每种模具的特征频谱（主频 Hz + 权重）；不同模具主频带明显不同
MOLD_SPECTRA = {
    "A": ([800.0, 1600.0], [1.0, 0.7]),
    "B": ([2600.0, 4200.0], [1.0, 0.8]),
    "C": ([500.0, 3400.0], [0.9, 1.0]),
}


def build_spectral_wav(
    impact_times: list[float],
    freqs: list[float],
    weights: list[float] | None = None,
    duration: float | None = None,
    sr: int = SR,
    gain: float = 0.8,
    leading_silence: float = 0.0,
    burst_ms: int = 55,
    noise_floor: float = 0.002,
    decay: float = 60.0,
    seed: int = 11,
) -> bytes:
    """生成冲击为「阻尼正弦叠加」的 WAV，冲击频谱由 freqs/weights 决定。

    与 build_waw 的白噪声冲击不同，这里每个冲击具有确定的频谱形状，
    可用于验证按冲击频谱辨认模具的逻辑；leading_silence 在整段前补零，
    gain 对整段做恒定缩放（模拟整体音量调高/调低）。
    """
    rng = np.random.default_rng(seed)
    if duration is None:
        duration = max(impact_times) + 0.4
    total_duration = duration + leading_silence
    n = int(round(total_duration * sr))
    x = rng.standard_normal(n).astype(np.float32) * noise_floor

    burst_n = int(burst_ms / 1000 * sr)
    tt = np.arange(burst_n) / sr
    if weights is None:
        weights = [1.0] * len(freqs)
    shape = np.zeros(burst_n, dtype=np.float32)
    for f, w in zip(freqs, weights):
        shape += w * np.sin(2 * np.pi * f * tt).astype(np.float32)
    shape *= np.exp(-decay * tt).astype(np.float32)
    peak = float(np.max(np.abs(shape)))
    shape = shape / peak if peak > 0 else shape

    for t in impact_times:
        s = int(round((t + leading_silence) * sr))
        e = min(n, s + burst_n)
        if s < n and e > s:
            x[s:e] += gain * shape[: e - s]

    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return buf.getvalue()


def post_identify(client, mold_ids, samples, unknown):
    """samples: {模具编号: [3 段 wav bytes]}，字段名与 API 约定一致。"""
    files = []
    for mold in mold_ids:
        for k, raw in enumerate(samples[mold], start=1):
            files.append((f"sample_{mold}_{k}", (f"{mold}_{k}.wav", raw, "audio/wav")))
    files.append(("unknown", ("unknown.wav", unknown, "audio/wav")))
    # dict + list 值可在 multipart 中正确表达重复字段（元组列表在当前
    # httpx/starlette 组合下会被吞掉，故不使用）
    data = {"mold_ids": list(mold_ids)}
    return client.post("/api/identify-molds", files=files, data=data)
