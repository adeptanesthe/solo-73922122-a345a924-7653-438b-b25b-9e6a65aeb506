"""按冲击频谱辨认模具，并以留一法（LOO）校准各模具的匹配范围。

设计要点
========
1. **冲击频谱指纹**：对每段录音中的每个冲击，在其时刻附近取固定长度
   短窗做 FFT，得到冲击幅度谱；每个冲击的谱先做 L2 归一化（整体音量
   被约去），同一录音内所有冲击的谱取平均后再次归一化，得到该录音的
   指纹向量。指纹只刻画频谱形状，因此：
   - 整体调高/调低音量（恒定增益）不改变归一化后的谱形；
   - 起始静音只平移冲击时刻，短窗以冲击时刻为基准截取，不受影响。
2. **相似度**：两指纹的余弦相似度（向量均已归一化，即点积）。
3. **留一法校准**：对每种模具轮流留出 1 段样音、用其余样音构成参考
   指纹去辨认它。留一过程中本模具各折的自相似度最小值/最大值构成该
   模具的「匹配范围」。只要有一段样音被错认或归属不明（与非本模具
   相似度差距过小），该模具 ``loo_passed=False`` —— 仍参与比较，但
   不能被推荐。
4. **辨认放行条件（全部满足才可推荐）**：
   a. 最佳相似度落在该模具留一校准的匹配范围内（下限允许 RANGE_MARGIN
      容差，高于上限视为更典型的本模具声音，照常接受）——与所有模具
      都不像时优先判为超范围，而非归属不明；
   b. 待辨录音在所有模具中唯一最佳（与第二名相似度差 > UNIQUE_MARGIN）；
   c. 该模具通过留一验证。
   任一不满足则辨认失败，返回明确原因，由前端保留当前参考。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audio import SAMPLE_RATE

# ---- 冲击频谱提取参数 ----
SPECTRUM_PRE_SEC = 0.005          # 冲击时刻向前取 5ms
SPECTRUM_POST_SEC = 0.050         # 向后取 50ms（覆盖主要衰减/振铃段）
NFFT = 1024                       # 零填充至 1024 点 FFT
F_MIN_HZ = 200.0                  # 有效频带下限（避开低频直流漂移）
F_MAX_HZ = 6000.0                 # 有效频带上限（16kHz 采样的有效辨识带）

# ---- 决策参数 ----
UNIQUE_MARGIN = 0.02              # 唯一最佳须领先第二名的最小相似度差
RANGE_MARGIN = 0.02               # 匹配范围下限允许的容差

_FREQS = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
_BAND = (_FREQS >= F_MIN_HZ) & (_FREQS <= F_MAX_HZ)


class IdentificationError(ValueError):
    """录音无法提取可用于辨认的冲击频谱。"""


@dataclass
class LooFold:
    """一次留一折的结果。"""
    held_out_index: int
    predicted_mold: str | None     # None 表示归属不明
    unique: bool
    correct: bool
    self_similarity: float
    scores: dict[str, float] = field(default_factory=dict)
    reason: str | None = None


@dataclass
class MoldCalibration:
    mold_id: str
    sample_count: int
    range_low: float
    range_high: float
    loo_passed: bool
    folds: list[LooFold] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        raise IdentificationError("冲击频谱能量为零，无法用于辨认")
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度；输入通常已归一化，这里仍做防御性归一化。"""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_impact_profile(
    samples: np.ndarray, impact_times: list[float]
) -> np.ndarray:
    """从一段录音的冲击时刻序列提取归一化冲击频谱指纹。

    每个冲击取固定短窗（Hann 窗 + 边缘零填充），FFT 后在有效频带内
    逐冲击 L2 归一化，再对全部冲击取平均并整体归一化。
    """
    if not impact_times:
        raise IdentificationError("录音中未检测到任何冲击，无法提取冲击频谱")

    pre = int(round(SPECTRUM_PRE_SEC * SAMPLE_RATE))
    post = int(round(SPECTRUM_POST_SEC * SAMPLE_RATE))
    win_len = pre + post
    hann = np.hanning(win_len).astype(np.float32)

    specs: list[np.ndarray] = []
    n_total = len(samples)
    for t in impact_times:
        center = int(round(t * SAMPLE_RATE))
        start = center - pre
        end = start + win_len
        seg = np.zeros(win_len, dtype=np.float32)
        src_lo = max(0, start)
        src_hi = min(n_total, end)
        if src_hi <= src_lo:
            continue
        seg[src_lo - start : src_hi - start] = samples[src_lo:src_hi]
        magnitude = np.abs(np.fft.rfft(seg * hann, n=NFFT))
        band_spec = magnitude[_BAND].astype(np.float32)
        norm = float(np.linalg.norm(band_spec))
        if norm > 1e-12:
            specs.append(band_spec / norm)

    if not specs:
        raise IdentificationError("冲击片段频谱能量为零，无法用于辨认")
    profile = np.mean(np.stack(specs, axis=0), axis=0)
    return _normalize(profile).astype(np.float32)


def _mean_profile(profiles: list[np.ndarray]) -> np.ndarray:
    return _normalize(np.mean(np.stack(profiles, axis=0), axis=0))


def _rank(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def calibrate_molds(
    profiles: dict[str, list[np.ndarray]],
) -> dict[str, MoldCalibration]:
    """留一法校准：每段样音轮流留出，用其余样音辨认。

    产出各模具的匹配范围（留一自相似度的最小/最大值）与是否通过验证。
    """
    # 各模具完整参考指纹（用全部样音），留一折中的「其他模具」参考用它
    full_refs = {mold: _mean_profile(plist) for mold, plist in profiles.items()}

    result: dict[str, MoldCalibration] = {}
    for mold, plist in profiles.items():
        k_total = len(plist)
        folds: list[LooFold] = []
        self_sims: list[float] = []
        failures: list[str] = []

        for k in range(k_total):
            held = plist[k]
            fold_refs = dict(full_refs)
            # 本模具折参考：仅用未留出的样音
            train = [p for j, p in enumerate(plist) if j != k]
            fold_refs[mold] = _mean_profile(train)

            scores = {
                other: cosine_similarity(held, ref)
                for other, ref in fold_refs.items()
            }
            ranking = _rank(scores)
            top_mold, top_score = ranking[0]
            runner_up = ranking[1] if len(ranking) > 1 else None
            unique = (
                runner_up is None
                or top_score - runner_up[1] > UNIQUE_MARGIN + 1e-12
            )
            self_sim = scores[mold]
            self_sims.append(self_sim)

            if not unique:
                tied = [
                    f"模具 {m}（相似度 {s:.3f}）"
                    for m, s in ranking[:2]
                ]
                reason = (
                    f"留出样音 {k + 1} 归属不明：{ '、'.join(tied) }"
                    f"相似度差距不超过 {UNIQUE_MARGIN:.2f}"
                )
                failures.append(reason)
                folds.append(
                    LooFold(
                        held_out_index=k,
                        predicted_mold=None,
                        unique=False,
                        correct=False,
                        self_similarity=self_sim,
                        scores=dict(scores),
                        reason=reason,
                    )
                )
                continue

            correct = top_mold == mold
            reason = None
            if not correct:
                reason = (
                    f"留出样音 {k + 1} 被错认为模具 {top_mold}"
                    f"（相似度 {top_score:.3f}），高于本模具相似度"
                    f" {self_sim:.3f}"
                )
                failures.append(reason)
            folds.append(
                LooFold(
                    held_out_index=k,
                    predicted_mold=top_mold,
                    unique=True,
                    correct=correct,
                    self_similarity=self_sim,
                    scores=dict(scores),
                    reason=reason,
                )
            )

        result[mold] = MoldCalibration(
            mold_id=mold,
            sample_count=k_total,
            range_low=float(min(self_sims)),
            range_high=float(max(self_sims)),
            loo_passed=not failures,
            folds=folds,
            failures=failures,
        )
    return result


def identify_mold(
    unknown_profile: np.ndarray,
    profiles: dict[str, list[np.ndarray]],
    calibrations: dict[str, MoldCalibration],
) -> dict[str, Any]:
    """对待辨录音执行辨认，返回 JSON 可序列化的决策详情。

    决策顺序：唯一最佳 → 落在匹配范围 → 该模具通过留一验证。
    """
    scores = {
        mold: cosine_similarity(unknown_profile, _mean_profile(plist))
        for mold, plist in profiles.items()
    }
    ranking = _rank(scores)
    top_mold, top_score = ranking[0]
    runner_up = ranking[1] if len(ranking) > 1 else None
    unique = (
        runner_up is None
        or top_score - runner_up[1] > UNIQUE_MARGIN + 1e-12
    )
    top_cal = calibrations[top_mold]
    # 高于上限说明比留一样本更典型，照常接受；下限给 RANGE_MARGIN 容差
    within_range = top_score >= top_cal.range_low - RANGE_MARGIN - 1e-12

    ranking_view = [
        {
            "mold_id": mold,
            "score": round(score, 4),
            "within_range": score
            >= calibrations[mold].range_low - RANGE_MARGIN - 1e-12,
            "loo_passed": calibrations[mold].loo_passed,
        }
        for mold, score in ranking
    ]

    calibration_view = {
        cal.mold_id: {
            "mold_id": cal.mold_id,
            "sample_count": cal.sample_count,
            "loo_passed": cal.loo_passed,
            "matching_range": {
                "low": round(cal.range_low, 4),
                "high": round(cal.range_high, 4),
            },
            "failures": list(cal.failures),
            "folds": [
                {
                    "held_out": f.held_out_index + 1,
                    "predicted_mold": f.predicted_mold,
                    "unique": f.unique,
                    "correct": f.correct,
                    "self_similarity": round(f.self_similarity, 4),
                    "scores": {m: round(s, 4) for m, s in f.scores.items()},
                    "reason": f.reason,
                }
                for f in cal.folds
            ],
        }
        for cal in calibrations.values()
    }

    base: dict[str, Any] = {
        "decision": "failed",
        "identified_mold": None,
        "unique": unique,
        "margin": round(
            top_score - runner_up[1] if runner_up is not None else 1.0, 4
        ),
        "scores": {m: round(s, 4) for m, s in scores.items()},
        "ranking": ranking_view,
        "calibration": calibration_view,
    }

    if not within_range:
        # 先判范围：与所有模具都不像（分数低于最佳模具的留一下限）时，
        # 即使两个低分模具彼此接近，也属于「不属于任何已建模模具」，
        # 而不是归属不明
        base.update(
            reason_code="out_of_range",
            reason=(
                f"待辨录音最接近模具 {top_mold}（相似度 {top_score:.3f}），"
                f"但低于该模具留一校准匹配范围下限"
                f" {top_cal.range_low:.3f}（容差 {RANGE_MARGIN:.2f}），"
                "不属于任何已建模模具，保留当前参考。"
            ),
        )
        return base

    if not unique:
        tied = "、".join(
            f"模具 {m}（{s:.3f}）" for m, s in ranking[:2]
        )
        base.update(
            reason_code="ambiguous",
            reason=(
                f"待辨录音归属不明：{tied}的冲击频谱相似度差距不超过"
                f" {UNIQUE_MARGIN:.2f}，无法唯一辨认，保留当前参考。"
            ),
        )
        return base

    if not top_cal.loo_passed:
        base.update(
            reason_code="mold_not_recommendable",
            reason=(
                f"待辨录音唯一匹配模具 {top_mold}，但该模具留一验证未通过"
                f"（{ '；'.join(top_cal.failures) }），按规则不可推荐，"
                "保留当前参考。"
            ),
        )
        return base

    base.update(
        decision="identified",
        identified_mold=top_mold,
        reason_code="unique_match",
        reason=(
            f"待辨录音唯一匹配模具 {top_mold}：相似度 {top_score:.3f}，"
            f"领先第二名 {base['margin']:.3f}，且该模具通过留一验证，"
            "可选取其样音填入核对页作为参考。"
        ),
    )
    return base
