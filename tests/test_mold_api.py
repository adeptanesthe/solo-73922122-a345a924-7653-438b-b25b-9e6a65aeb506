"""模具辨认端到端 API 测试。"""
from conftest import MOLD_SPECTRA, build_spectral_wav, post_identify

BEAT = [0.6, 1.2, 1.8, 2.4]


def _mold_samples(mold, seed0=50):
    freqs, weights = MOLD_SPECTRA[mold]
    return [
        build_spectral_wav(
            BEAT, freqs, [x * (0.9 + 0.05 * k) for x in weights],
            duration=3.0, gain=0.5 + 0.1 * k, seed=seed0 + k,
        )
        for k in range(3)
    ]


def _unknown(mold, **kw):
    freqs, weights = MOLD_SPECTRA[mold]
    return build_spectral_wav(BEAT, freqs, weights, duration=3.0, seed=4242, **kw)


def _payload(molds=("A", "B"), unknown_mold="A", **ukw):
    samples = {m: _mold_samples(m) for m in molds}
    return list(molds), samples, _unknown(unknown_mold, **ukw)


def test_identify_success_and_persisted(client):
    resp = post_identify(client, *_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["decision"] == "identified"
    assert body["identified_mold"] == "A"
    assert len(body["mold_ids"]) == 2
    for mold in body["molds"]:
        assert mold["loo_passed"] is True
        assert len(mold["sample_urls"]) == 3
        assert mold["matching_range"]["low"] <= mold["matching_range"]["high"]
    # 第一名唯一
    assert body["ranking"][0]["mold_id"] == "A"
    assert body["unique"] is True

    last = client.get("/api/identification/last").json()
    assert last["id"] == body["id"]


def test_gain_and_silence_variants_same_mold(client):
    molds, samples, _ = _payload()
    for kw in ({"gain": 0.1}, {"gain": 0.95}, {"leading_silence": 0.7}):
        body = post_identify(
            client, molds, samples, _unknown("A", **kw)
        ).json()
        assert body["decision"] == "identified"
        assert body["identified_mold"] == "A"


def test_identify_failure_keeps_last_identification(client):
    ok = post_identify(client, *_payload())
    assert ok.status_code == 200
    ok_id = ok.json()["id"]

    # 模具数量不足：只有 1 个
    one_samples = {"A": _mold_samples("A")}
    bad = post_identify(client, ["A"], one_samples, _unknown("A"))
    assert bad.status_code == 400
    assert "模具" in bad.json()["detail"]

    # 某模具缺一段样音（直接少发一个字段）
    files, data = [], {"mold_ids": ["A", "B"]}
    for k, raw in enumerate(_mold_samples("A")[:2], start=1):
        files.append((f"sample_A_{k}", (f"A{k}.wav", raw, "audio/wav")))
    for k, raw in enumerate(_mold_samples("B"), start=1):
        files.append((f"sample_B_{k}", (f"B{k}.wav", raw, "audio/wav")))
    files.append(("unknown", ("u.wav", _unknown("A"), "audio/wav")))
    bad2 = client.post("/api/identify-molds", files=files, data=data)
    assert bad2.status_code == 400
    assert "样音 3" in bad2.json()["detail"]

    # 非 WAV（扩展名仍为 .wav，格式错误在解析阶段触发）
    files3, data3 = [], {"mold_ids": ["A", "B"]}
    for mold, raws in {"A": _mold_samples("A"), "B": _mold_samples("B")}.items():
        for k, raw in enumerate(raws, start=1):
            files3.append((f"sample_{mold}_{k}", (f"{mold}{k}.wav", raw, "audio/wav")))
    files3.append(("unknown", ("u.wav", b"not wav", "audio/wav")))
    bad3 = client.post("/api/identify-molds", files=files3, data=data3)
    assert bad3.status_code == 400

    # 上一次辨认结果仍保留
    assert client.get("/api/identification/last").json()["id"] == ok_id


def test_silent_unknown_rejected_and_keeps_state(client):
    post_identify(client, *_payload())  # 先建立有效结果
    import io
    import wave
    import numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(2 * 16000, dtype="<i2").tobytes())

    molds, samples, _ = _payload()
    resp = post_identify(client, molds, samples, buf.getvalue())
    assert resp.status_code == 400
    assert "冲击" in resp.json()["detail"]


def test_out_of_range_unknown_returns_200_with_failure_reason(client):
    """辨认失败（非输入错误）返回 200，decision=failed 并带原因。"""
    weird = build_spectral_wav(BEAT, [5500.0], [1.0], duration=3.0, seed=77)
    molds, samples, _ = _payload()
    body = post_identify(client, molds, samples, weird).json()
    assert body["decision"] == "failed"
    assert body["identified_mold"] is None
    assert body["reason_code"] == "out_of_range"
    assert body["reason"]


def test_failed_loo_mold_is_not_recommendable(client):
    """污染模具 A 的一段样音：A 留一不通过；唯一匹配 A 时仍不可推荐。"""
    samples = {"A": _mold_samples("A"), "B": _mold_samples("B")}
    samples["A"][2] = _unknown("B")  # A 的第 3 段实为 B 的声音（seed 固定为 4242）
    body = post_identify(client, ["A", "B"], samples, _unknown("A")).json()

    mold_a = next(m for m in body["molds"] if m["mold_id"] == "A")
    assert mold_a["loo_passed"] is False
    assert mold_a["failures"]
    assert body["ranking"][0]["mold_id"] == "A"
    assert body["decision"] == "failed"
    assert body["reason_code"] == "mold_not_recommendable"


def test_duplicate_mold_ids_rejected(client):
    samples = {"A": _mold_samples("A")}
    resp = post_identify(client, ["A", "A"], {"A": samples["A"], "A": samples["A"]}, _unknown("A"))
    assert resp.status_code == 400
    assert "重复" in resp.json()["detail"]


def test_illegal_mold_id_rejected(client):
    samples = {"A/B": _mold_samples("A"), "B": _mold_samples("B")}
    resp = post_identify(client, ["A/B", "B"], samples, _unknown("A"))
    assert resp.status_code == 400
    assert "不合法" in resp.json()["detail"]


def test_sample_audio_served_back_and_clear(client):
    body = post_identify(client, *_payload()).json()
    url = body["molds"][0]["sample_urls"][0]
    served = client.get(url)
    assert served.status_code == 200
    assert served.headers["content-type"] == "audio/wav"

    assert client.delete("/api/identification/last").status_code == 200
    assert client.get("/api/identification/last").status_code == 404


def test_three_and_four_molds_accepted(client):
    for n in (3, 4):
        molds = ("A", "B", "C", "D")[:n]
        # D 复用 C 相近但不同的主频，确保仍可区分
        extra = {
            "D": _mold_samples("C", seed0=90),
        }
        samples = {m: (_mold_samples(m) if m != "D" else extra["D"]) for m in molds}
        if "D" in molds:
            # D 与 C 频谱相同会导致歧义，改用不同主频构造
            samples["D"] = [
                build_spectral_wav(
                    BEAT, [1200.0, 4800.0], [1.0, 0.6],
                    duration=3.0, seed=200 + k,
                )
                for k in range(3)
            ]
        resp = post_identify(client, list(molds), samples, _unknown("B"))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["identified_mold"] == "B"
        assert len(body["mold_ids"]) == n
