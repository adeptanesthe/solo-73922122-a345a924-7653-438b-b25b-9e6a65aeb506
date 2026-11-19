"""模具辨认（冲击频谱指纹、留一校准、放行决策）测试。"""
import numpy as np
import pytest

from app.audio import detect_impacts, parse_wav
from app.identification import (
    IdentificationError,
    calibrate_molds,
    cosine_similarity,
    extract_impact_profile,
    identify_mold,
)
from conftest import MOLD_SPECTRA, build_spectral_wav

# 两种模具节拍完全一样，仅冲击频谱不同
BEAT = [0.6, 1.2, 1.8, 2.4]


def _profiles_for_molds(molds, seed0=100):
    """合成每种模具 3 段样音并提取指纹（轻微扰动保证三段不完全相同）。"""
    profiles = {}
    raws = {}
    for mi, mold in enumerate(molds):
        freqs, weights = MOLD_SPECTRA[mold]
        plist, raw_list = [], []
        for k in range(3):
            w = [x * (0.92 + 0.05 * k) for x in weights]
            raw = build_spectral_wav(
                BEAT, freqs, w, duration=3.0,
                gain=0.5 + 0.1 * k, seed=seed0 + mi * 7 + k,
            )
            samples = parse_wav(raw)
            det = detect_impacts(samples)
            assert len(det.times) == 4
            plist.append(extract_impact_profile(samples, det.times))
            raw_list.append(raw)
        profiles[mold] = plist
        raws[mold] = raw_list
    return profiles, raws


def _unknown_of(mold, **kw):
    freqs, weights = MOLD_SPECTRA[mold]
    raw = build_spectral_wav(BEAT, freqs, weights, duration=3.0, seed=999, **kw)
    samples = parse_wav(raw)
    return extract_impact_profile(samples, detect_impacts(samples).times), raw


def test_gain_and_leading_silence_do_not_change_identity():
    """整体音量调高/调低、加起始静音后，归属必须相同。"""
    profiles, _ = _profiles_for_molds(["A", "B"])
    cals = calibrate_molds(profiles)

    base, _ = _unknown_of("A")
    quiet, _ = _unknown_of("A", gain=0.1)            # 调低整体音量
    loud, _ = _unknown_of("A", gain=0.95)            # 调高整体音量
    shifted, _ = _unknown_of("A", leading_silence=0.7)  # 起始静音

    base_dec = identify_mold(base, profiles, cals)
    assert base_dec["identified_mold"] == "A"
    for p in (quiet, loud, shifted):
        dec = identify_mold(p, profiles, cals)
        assert dec["identified_mold"] == "A"
        assert dec["decision"] == "identified"
        # 归一化后相似度差异应很小（音量/静音不改变频谱形状）
        assert cosine_similarity(p, base) == pytest.approx(1.0, abs=0.02)


def test_loo_calibration_passes_for_distinct_molds():
    profiles, _ = _profiles_for_molds(["A", "B", "C"])
    cals = calibrate_molds(profiles)
    for mold, cal in cals.items():
        assert cal.loo_passed is True
        assert len(cal.folds) == 3
        assert all(f.correct and f.unique for f in cal.folds)
        # 匹配范围由留一自相似度的最小/最大值构成
        sims = [f.self_similarity for f in cal.folds]
        assert cal.range_low == pytest.approx(min(sims), abs=1e-9)
        assert cal.range_high == pytest.approx(max(sims), abs=1e-9)


def test_misclassified_loo_sample_blocks_recommendation():
    """留一时某模具样音被错认：仍参与比较，但不能被推荐。"""
    profiles, _ = _profiles_for_molds(["A", "B"])
    # 把模具 A 的第 3 段样音替换成模具 B 的频谱 -> 该折必被错认为 B
    b_shape, _ = _unknown_of("B")
    profiles["A"][2] = b_shape

    cals = calibrate_molds(profiles)
    assert cals["A"].loo_passed is False
    bad_fold = next(f for f in cals["A"].folds if f.held_out_index == 2)
    assert bad_fold.correct is False
    assert bad_fold.predicted_mold == "B"
    assert cals["A"].failures  # 有可读的失败原因
    # B 模具本身样音未受污染，仍通过
    assert cals["B"].loo_passed is True

    unknown_a, _ = _unknown_of("A")
    dec = identify_mold(unknown_a, profiles, cals)
    # A 仍是唯一最佳且在范围内，但因留一未通过不可推荐
    assert dec["ranking"][0]["mold_id"] == "A"
    assert dec["decision"] == "failed"
    assert dec["reason_code"] == "mold_not_recommendable"
    assert dec["identified_mold"] is None


def test_tie_between_molds_is_ambiguous_failure():
    """两个模具频谱过于接近、相似度差距不足时，归属不明。"""
    profiles, _ = _profiles_for_molds(["A", "B"])
    # 将 B 的指纹直接改成与 A 相同，构造相似度并列
    profiles["B"] = [p.copy() for p in profiles["A"]]
    cals = calibrate_molds(profiles)
    assert all(not cal.loo_passed for cal in cals.values())

    unknown = profiles["A"][0]
    dec = identify_mold(unknown, profiles, cals)
    assert dec["decision"] == "failed"
    assert dec["reason_code"] == "ambiguous"
    assert dec["unique"] is False


def test_unknown_spectrum_outside_all_ranges_fails():
    """待辨录音与所有模具都不像：超出匹配范围，辨认失败并说明原因。"""
    profiles, _ = _profiles_for_molds(["A", "B"])
    cals = calibrate_molds(profiles)
    raw = build_spectral_wav(
        BEAT, [5500.0], [1.0], duration=3.0, seed=31337
    )
    samples = parse_wav(raw)
    unknown = extract_impact_profile(samples, detect_impacts(samples).times)

    dec = identify_mold(unknown, profiles, cals)
    assert dec["decision"] == "failed"
    assert dec["reason_code"] == "out_of_range"
    assert "匹配范围" in dec["reason"]


def test_extract_profile_requires_impacts():
    silent = np.zeros(2 * 16000, dtype=np.float32)
    with pytest.raises(IdentificationError):
        extract_impact_profile(silent, [])


def test_profiles_are_normalized_and_gain_invariant():
    raw1 = build_spectral_wav(BEAT, *MOLD_SPECTRA["A"], duration=3.0, gain=0.1)
    raw2 = build_spectral_wav(BEAT, *MOLD_SPECTRA["A"], duration=3.0, gain=0.9)
    p1 = extract_impact_profile(parse_wav(raw1), detect_impacts(parse_wav(raw1)).times)
    p2 = extract_impact_profile(parse_wav(raw2), detect_impacts(parse_wav(raw2)).times)
    assert np.linalg.norm(p1) == pytest.approx(1.0, abs=1e-6)
    # 仅增益不同：频谱形状几乎一致（噪声地板带来极小差异）
    assert cosine_similarity(p1, p2) == pytest.approx(1.0, abs=0.01)
