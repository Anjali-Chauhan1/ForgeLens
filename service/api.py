"""FastAPI service implementing the /v1/analyze surface from the
blueprint's API-shape section. Models are loaded lazily and optionally:
if no trained checkpoints are present the service still runs end-to-end
using the classical (zero-ML) forensics branches plus the documented
fallback fusion weighting, so the system is testable before any
training has completed.

Async jobs use an in-process queue here for simplicity; swap
`JOB_STORE`/`run_analysis` for Celery + Redis per the blueprint's stack
recommendation once you need horizontal scaling.

API-key auth is OFF by default (no FORGELENS_API_KEYS set) so local
development and the test suite don't need credentials; set
FORGELENS_API_KEYS="key1,key2" to require the X-API-Key header in
any real deployment.
"""
from __future__ import annotations
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import Response

from forgelens.inference.pipeline import ForgeLensPipeline
from forgelens.report.evidence import to_result_dict, narrative_summary
from forgelens.report.pdf import render_report_pdf
from forgelens.fusion.fuse import FusionModel
from forgelens.provenance.retrieval import PerceptualHashIndex
from service.schemas import AnalyzeResult, JobStatus
from service.auth import require_api_key, record_audit_event

app = FastAPI(title="ForgeLens", version="0.1.0")

MODEL_DIR = os.environ.get("FORGELENS_MODEL_DIR", "runs")
RETRIEVAL_INDEX_PATH = os.environ.get("FORGELENS_RETRIEVAL_INDEX")  # optional
SYNC_SIZE_THRESHOLD_PX = 1600 * 1600
MAX_BATCH_SIZE = int(os.environ.get("FORGELENS_MAX_BATCH", 20))

JOB_STORE: dict[str, dict] = {}
REPORT_CACHE: dict[str, object] = {}  # sha256 -> AnalysisReport (for PDF export)


def _load_pipeline() -> ForgeLensPipeline:
    localization_model = None
    synthetic_model = None
    fusion_model = FusionModel()

    loc_ckpt = Path(MODEL_DIR) / "localize" / "best.pt"
    if loc_ckpt.exists():
        import torch
        from forgelens.models.backbone import LocalizationNet
        localization_model = LocalizationNet()
        state = torch.load(loc_ckpt, map_location="cpu")
        # Default to raw weights, not EMA -- see the detailed comment in
        # scripts/evaluate_test_set.py's load_trained_pipeline(). EMA
        # only helps once a run is long enough for its warmup schedule
        # to converge; on a short/early checkpoint it can be dominated
        # by the near-random initialization and silently under-perform
        # the weights that were actually validated as "best".
        use_ema = os.environ.get("FORGELENS_USE_EMA", "").lower() in ("1", "true", "yes")
        localization_model.load_state_dict(state["ema"] if use_ema else state["model"])
        localization_model.eval()

    fusion_json = Path(MODEL_DIR) / "fusion" / "fusion_model.json"
    if fusion_json.exists():
        fusion_model = FusionModel.load(str(Path(MODEL_DIR) / "fusion" / "fusion_model"))

    retrieval_index = None
    if RETRIEVAL_INDEX_PATH:
        retrieval_index = PerceptualHashIndex(store_path=RETRIEVAL_INDEX_PATH)

    return ForgeLensPipeline(
        localization_model=localization_model,
        synthetic_model=synthetic_model,
        fusion_model=fusion_model,
        retrieval_index=retrieval_index,
    )


_pipeline: ForgeLensPipeline | None = None


def get_pipeline() -> ForgeLensPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_pipeline()
    return _pipeline


def _analyze_path(path: str, key_hash: str = "unauthenticated") -> dict:
    pipeline = get_pipeline()
    report = pipeline.analyze(path)
    result = to_result_dict(report)
    result["narrative"] = narrative_summary(report)
    REPORT_CACHE[result["sha256"]] = report
    record_audit_event("analyze", result["sha256"], key_hash,
                        detail={"verdict": result["verdict"], "abstained": result["abstained"]})
    return result


def _run_job(job_id: str, path: str, key_hash: str):
    try:
        result = _analyze_path(path, key_hash)
        JOB_STORE[job_id] = {"status": "done", "result": result, "error": None}
    except Exception as e:  # surfaced to the client, not swallowed
        JOB_STORE[job_id] = {"status": "failed", "result": None, "error": str(e)}
        record_audit_event("analyze_failed", None, key_hash, detail={"error": str(e)})
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _save_upload(image: UploadFile, content: bytes) -> str:
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return tmp_path


@app.post("/v1/analyze")
async def analyze(background_tasks: BackgroundTasks, image: UploadFile = File(...),
                   key_hash: str = Depends(require_api_key)):
    content = await image.read()
    tmp_path = _save_upload(image, content)

    from PIL import Image as PILImage
    with PILImage.open(tmp_path) as im:
        pixels = im.width * im.height

    if pixels <= SYNC_SIZE_THRESHOLD_PX:
        try:
            result = _analyze_path(tmp_path, key_hash)
        finally:
            os.unlink(tmp_path)
        return result

    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = {"status": "queued", "result": None, "error": None}
    record_audit_event("analyze_queued", None, key_hash, detail={"job_id": job_id})
    background_tasks.add_task(_run_job, job_id, tmp_path, key_hash)
    return {"job_id": job_id, "status": "queued"}


@app.post("/v1/analyze/batch")
async def analyze_batch(background_tasks: BackgroundTasks, images: list[UploadFile] = File(...),
                         key_hash: str = Depends(require_api_key)):
    """Batch upload / folder analysis, per the blueprint's Core platform
    feature list. Every image is queued as an independent async job so
    one slow/corrupt file in a batch doesn't block the rest; the caller
    polls each job_id via the existing /v1/analyze/{job_id} endpoint.
    """
    if len(images) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413,
                             detail=f"Batch too large: {len(images)} > {MAX_BATCH_SIZE}. "
                                    f"Split into smaller batches.")

    job_ids = []
    for image in images:
        content = await image.read()
        tmp_path = _save_upload(image, content)
        job_id = str(uuid.uuid4())
        JOB_STORE[job_id] = {"status": "queued", "result": None, "error": None,
                              "filename": image.filename}
        background_tasks.add_task(_run_job, job_id, tmp_path, key_hash)
        job_ids.append({"job_id": job_id, "filename": image.filename})

    record_audit_event("batch_analyze_queued", None, key_hash,
                        detail={"count": len(job_ids)})
    return {"jobs": job_ids, "count": len(job_ids)}


@app.get("/v1/analyze/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, key_hash: str = Depends(require_api_key)):
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"job_id": job_id, **{k: v for k, v in job.items() if k != "filename"}}


@app.get("/v1/report/{sha256}.pdf")
async def get_report_pdf(sha256: str, key_hash: str = Depends(require_api_key)):
    report = REPORT_CACHE.get(sha256)
    if report is None:
        raise HTTPException(status_code=404,
                             detail="No cached analysis for this hash. Re-run /v1/analyze.")
    pdf_bytes = render_report_pdf(report, image_path=f"sha256:{sha256}")
    record_audit_event("report_pdf_exported", sha256, key_hash)
    return Response(content=pdf_bytes, media_type="application/pdf",
                     headers={"Content-Disposition": f'attachment; filename="{sha256[:12]}.pdf"'})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
