"""FastAPI 入口：上传参考/待检 WAV，返回冲次核对结果。"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile as StarletteUploadFile

from .audio import AudioValidationError, MAX_DURATION_SEC, SAMPLE_RATE
from .identification import IdentificationError
from .service import (
    MAX_UPLOAD_BYTES,
    MOLD_ID_MAX,
    MOLD_ID_MIN,
    SAMPLES_PER_MOLD,
    Store,
)

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/workspace/data"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="冲压机冲次声音核对", version="1.0.0")
store = Store(DATA_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/data/uploads/{name}")
def get_upload(name: str) -> FileResponse:
    # 动态读取 store 的上传目录，更换 DATA_DIR 或测试夹具后仍可回传录音
    if "/" in name or ".." in name or not name.endswith(".wav"):
        raise HTTPException(status_code=400, detail="非法文件名")
    path = store.uploads_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="录音不存在")
    return FileResponse(path, media_type="audio/wav")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/result/last")
def last_result() -> dict:
    result = store.get_last()
    if result is None:
        raise HTTPException(status_code=404, detail="暂无有效检查结果")
    return result


@app.delete("/api/result/last")
def clear_result() -> dict[str, str]:
    store.clear()
    return {"status": "cleared"}


@app.post("/api/inspect")
async def inspect(
    reference: UploadFile = File(..., description="正常循环参考 WAV"),
    test: UploadFile = File(..., description="待检循环 WAV"),
) -> dict:
    async def read_wav(file: UploadFile, label: str) -> bytes:
        if not (file.filename or "").lower().endswith(".wav"):
            raise HTTPException(
                status_code=400, detail=f"{label}必须是 .wav 文件"
            )
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"{label}文件为空")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{label}文件过大；仅支持 {SAMPLE_RATE}Hz/16bit/单声道、"
                    f"不超过 {MAX_DURATION_SEC:.0f} 秒的 WAV"
                ),
            )
        return raw

    ref_raw = await read_wav(reference, "参考录音")
    test_raw = await read_wav(test, "待检录音")

    try:
        # 任何 AudioValidationError 都会在 store 写入前抛出，
        # 因此无效音频不会覆盖上一次有效结果
        return store.inspect(ref_raw, test_raw)
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/identification/last")
def last_identification() -> dict:
    result = store.get_last_identification()
    if result is None:
        raise HTTPException(status_code=404, detail="暂无模具辨认结果")
    return result


@app.delete("/api/identification/last")
def clear_identification() -> dict[str, str]:
    store.clear_identification()
    return {"status": "cleared"}


def _read_mold_file(upload: UploadFile, label: str) -> bytes:
    if not (upload.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail=f"{label}必须是 .wav 文件")
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{label}文件为空")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label}文件过大；仅支持 {SAMPLE_RATE}Hz/16bit/单声道、"
                f"不超过 {MAX_DURATION_SEC:.0f} 秒的 WAV"
            ),
        )
    return raw


@app.post("/api/identify-molds")
async def identify_molds(request: Request) -> dict:
    """模具辨认：2~4 个模具、每模具 3 段样音 + 1 段待辨完整循环。

    表单约定：
    - 可重复字段 ``mold_ids``：模具编号（2~4 个，不可重复）
    - 样音字段名 ``sample_<模具编号>_<1..3>``
    - 待辨录音字段 ``unknown``
    """
    form = await request.form()
    mold_ids = form.getlist("mold_ids")
    if not (MOLD_ID_MIN <= len(mold_ids) <= MOLD_ID_MAX):
        raise HTTPException(
            status_code=400,
            detail=f"需输入 {MOLD_ID_MIN}～{MOLD_ID_MAX} 个模具编号，"
            f"当前为 {len(mold_ids)} 个",
        )
    if len(set(mold_ids)) != len(mold_ids):
        raise HTTPException(status_code=400, detail="模具编号存在重复，请去重后重试")
    if any(not str(m).strip() for m in mold_ids):
        raise HTTPException(status_code=400, detail="模具编号不能为空")

    samples: dict[str, list[bytes]] = {}
    for mold in mold_ids:
        mold = str(mold).strip()
        raws: list[bytes] = []
        for k in range(1, SAMPLES_PER_MOLD + 1):
            upload = form.get(f"sample_{mold}_{k}")
            if not isinstance(upload, StarletteUploadFile):
                raise HTTPException(
                    status_code=400,
                    detail=f"模具 {mold} 的样音 {k} 缺失，"
                    f"每个模具需上传 {SAMPLES_PER_MOLD} 段样音",
                )
            raws.append(_read_mold_file(upload, f"模具 {mold} 样音 {k}"))
        samples[mold] = raws

    unknown = form.get("unknown")
    if not isinstance(unknown, StarletteUploadFile):
        raise HTTPException(status_code=400, detail="缺少待辨认完整循环录音")
    unknown_raw = _read_mold_file(unknown, "待辨认录音")

    try:
        # 解析/检测/辨认任一步失败都在 store 写入前抛出，
        # 上一次有效辨认结果与核对页参考均原样保留
        return store.identify_molds(mold_ids, samples, unknown_raw)
    except (AudioValidationError, IdentificationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
