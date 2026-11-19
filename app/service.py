"""检查服务与结果存储。

无效音频（格式错误、参考不达标等）直接抛错，由 API 层返回 400，
且完全不触碰已保存的上一次有效结果。
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import (
    detect_impacts,
    parse_wav,
    validate_reference,
)
from .identification import (
    IdentificationError,
    calibrate_molds,
    extract_impact_profile,
    identify_mold,
)
from .matching import align_impacts

MAX_UPLOAD_BYTES = 16_000 * 2 * 8 + 4096  # 8 秒 16k/16bit/单声道 + WAV 头余量
MOLD_ID_MIN = 2
MOLD_ID_MAX = 4
MOLD_ID_NAME_MAX = 32
SAMPLES_PER_MOLD = 3
_MOLD_ID_BAD = set('/\\:*?"<>| \t\n\r')


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.uploads_dir = data_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_result: dict[str, Any] | None = None
        self._last_identification: dict[str, Any] | None = None

    def _save_wav(self, raw: bytes, prefix: str) -> str:
        digest = hashlib.sha256(raw).hexdigest()[:16]
        name = f"{prefix}_{digest}.wav"
        path = self.uploads_dir / name
        if not path.exists():
            path.write_bytes(raw)
        return f"/data/uploads/{name}"

    def inspect(self, reference_raw: bytes, test_raw: bytes) -> dict[str, Any]:
        # 先完整解析与检测，任何一步抛错都不会写入状态
        ref_samples = parse_wav(reference_raw)
        test_samples = parse_wav(test_raw)

        ref_det = detect_impacts(ref_samples)
        validate_reference(ref_det)  # 参考不达标 -> AudioValidationError
        test_det = detect_impacts(test_samples)

        alignment = align_impacts(ref_det.times, test_det.times)

        matched_ref = {i for i, _ in alignment.matched}
        matched_test = {j for _, j in alignment.matched}
        ref_url = self._save_wav(reference_raw, "ref")
        test_url = self._save_wav(test_raw, "test")

        reference_impacts = [
            {
                "index": i,
                "time": round(t, 4),
                "matched": i in matched_ref,
            }
            for i, t in enumerate(ref_det.times)
        ]
        test_impacts = [
            {
                "index": j,
                "time": round(t, 4),
                "matched": j in matched_test,
                "is_extra": j in alignment.extra_test_indices,
            }
            for j, t in enumerate(test_det.times)
        ]

        missed = [
            {
                "ref_index": i,
                "ref_time": round(ref_det.times[i], 4),
                "expected_time": round(
                    alignment.expected_missed_times.get(i, ref_det.times[i]), 4
                ),
            }
            for i in alignment.missed_ref_indices
        ]
        extras = [
            {
                "test_index": j,
                "time": round(test_det.times[j], 4),
            }
            for j in alignment.extra_test_indices
        ]
        matches = [
            {
                "ref_index": i,
                "test_index": j,
                "ref_time": round(ref_det.times[i], 4),
                "test_time": round(test_det.times[j], 4),
            }
            for i, j in alignment.matched
        ]

        verdict = "ok"
        if alignment.ambiguous:
            verdict = "ambiguous"
        elif alignment.miss_count or alignment.extra_count:
            verdict = "defect"

        warnings: list[str] = []
        if not test_det.times:
            warnings.append("待检录音未检测到任何冲击，全部参考冲次按漏冲处理")
        if test_det.suppressed_echoes:
            warnings.append(
                f"已抑制 {len(test_det.suppressed_echoes)} 个 40ms 内回声，不计多冲"
            )
        if alignment.ambiguous:
            warnings.append("存在多种同样符合节奏与最少漏/多冲规则的归属，请人工复听")

        result = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "ambiguous": alignment.ambiguous,
            "optimal_way_count": alignment.optimal_way_count,
            "summary": {
                "reference_impact_count": len(ref_det.times),
                "test_impact_count": len(test_det.times),
                "matched_count": len(alignment.matched),
                "miss_count": alignment.miss_count,
                "extra_count": alignment.extra_count,
                "suppressed_echo_count": len(test_det.suppressed_echoes),
            },
            "reference": {
                "url": ref_url,
                "duration": round(ref_det.duration, 4),
                "impacts": reference_impacts,
            },
            "test": {
                "url": test_url,
                "duration": round(test_det.duration, 4),
                "impacts": test_impacts,
            },
            "matches": matches,
            "missed": missed,
            "extras": extras,
            "timing_ratios": [round(r, 4) for r in alignment.timing_ratios],
            "warnings": warnings,
        }

        with self._lock:
            self._last_result = result
        return result

    def get_last(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._last_result is None else dict(self._last_result)

    def clear(self) -> None:
        with self._lock:
            self._last_result = None

    # ---------------- 模具辨认 ----------------

    @staticmethod
    def _validate_mold_payload(
        mold_ids: list[str], samples: dict[str, list[bytes]], unknown_raw: bytes
    ) -> None:
        if not (MOLD_ID_MIN <= len(mold_ids) <= MOLD_ID_MAX):
            raise IdentificationError(
                f"需输入 {MOLD_ID_MIN}～{MOLD_ID_MAX} 个模具编号，"
                f"当前为 {len(mold_ids)} 个"
            )
        if len(set(mold_ids)) != len(mold_ids):
            raise IdentificationError("模具编号存在重复，请去重后重试")
        for mold in mold_ids:
            if not mold or not mold.strip():
                raise IdentificationError("模具编号不能为空")
            if len(mold) > MOLD_ID_NAME_MAX or any(
                ch in _MOLD_ID_BAD for ch in mold
            ):
                raise IdentificationError(
                    f"模具编号「{mold}」不合法：长度不超过 {MOLD_ID_NAME_MAX} "
                    "个字符，且不能包含路径分隔符、空白或 :*?\"<>|"
                )
        for mold in mold_ids:
            n = len(samples.get(mold, []))
            if n != SAMPLES_PER_MOLD:
                raise IdentificationError(
                    f"模具 {mold} 需上传 {SAMPLES_PER_MOLD} 段已确认正常的"
                    f"样音，当前为 {n} 段"
                )
        if not unknown_raw:
            raise IdentificationError("待辨认完整循环录音为空")

    @staticmethod
    def _safe_prefix(mold: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in mold)

    def identify_molds(
        self,
        mold_ids: list[str],
        samples: dict[str, list[bytes]],
        unknown_raw: bytes,
    ) -> dict[str, Any]:
        """完整的模具辨认流程；任何校验失败都不写入已保存的辨认结果。"""
        self._validate_mold_payload(mold_ids, samples, unknown_raw)

        # 1. 全部解析与指纹提取，任一段不合格直接抛错，不触碰已存结果
        profiles: dict[str, list] = {}
        saved_samples: dict[str, list[str]] = {}
        for mold in mold_ids:
            mold_profiles = []
            mold_urls = []
            for idx, raw in enumerate(samples[mold]):
                wav_samples = parse_wav(raw)
                det = detect_impacts(wav_samples)
                if not det.times:
                    raise IdentificationError(
                        f"模具 {mold} 的样音 {idx + 1} 未检测到任何冲击，"
                        "无法提取冲击频谱，请更换样音"
                    )
                mold_profiles.append(extract_impact_profile(wav_samples, det.times))
                mold_urls.append(
                    self._save_wav(raw, f"mold_{self._safe_prefix(mold)}_{idx + 1}")
                )
            profiles[mold] = mold_profiles
            saved_samples[mold] = mold_urls

        unknown_samples = parse_wav(unknown_raw)
        unknown_det = detect_impacts(unknown_samples)
        if not unknown_det.times:
            raise IdentificationError(
                "待辨认录音未检测到任何冲击，无法提取冲击频谱"
            )
        unknown_profile = extract_impact_profile(
            unknown_samples, unknown_det.times
        )
        unknown_url = self._save_wav(unknown_raw, "mold_unknown")

        # 2. 留一法校准各模具匹配范围
        calibrations = calibrate_molds(profiles)

        # 3. 待辨录音对所有模具比较并做出放行/失败决策
        decision = identify_mold(unknown_profile, profiles, calibrations)

        result = {
            "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision["decision"],
            "identified_mold": decision["identified_mold"],
            "reason": decision["reason"],
            "reason_code": decision["reason_code"],
            "unique": decision["unique"],
            "margin": decision["margin"],
            "mold_ids": list(mold_ids),
            "unknown": {
                "url": unknown_url,
                "duration": round(unknown_det.duration, 4),
                "impact_count": len(unknown_det.times),
            },
            "molds": [
                {
                    "mold_id": mold,
                    "sample_urls": saved_samples[mold],
                    "loo_passed": calibrations[mold].loo_passed,
                    "matching_range": {
                        "low": round(calibrations[mold].range_low, 4),
                        "high": round(calibrations[mold].range_high, 4),
                    },
                    "failures": list(calibrations[mold].failures),
                }
                for mold in mold_ids
            ],
            "ranking": decision["ranking"],
            "scores": decision["scores"],
            "calibration": decision["calibration"],
        }
        with self._lock:
            self._last_identification = result
        return result

    def get_last_identification(self) -> dict[str, Any] | None:
        with self._lock:
            return (
                None
                if self._last_identification is None
                else dict(self._last_identification)
            )

    def clear_identification(self) -> None:
        with self._lock:
            self._last_identification = None
