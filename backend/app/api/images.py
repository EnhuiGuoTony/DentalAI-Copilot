import shutil
import uuid
from pathlib import Path
from uuid import UUID

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import ClinicalCase, XrayFinding, XrayImage
from app.db.session import get_db
from app.schemas.xray import FindingReviewUpdate, XrayFindingRead, XrayImageRead
from app.services.vision_service import VisionService

router = APIRouter(prefix="", tags=["images"])


@router.post("/cases/{case_id}/xray", response_model=XrayImageRead)
def upload_xray(case_id: UUID, file: UploadFile = File(...), db: Session = Depends(get_db)):
    case = db.get(ClinicalCase, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    settings = get_settings()
    image_id = uuid.uuid4()
    ext = Path(file.filename or "xray.png").suffix.lower() or ".png"
    target_dir = Path(settings.upload_dir) / str(case.patient_id) / str(case_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{image_id}{ext}"

    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    img = cv2.imread(str(target))
    height = width = None
    if img is not None:
        height, width = img.shape[:2]

    image = XrayImage(id=image_id, patient_id=case.patient_id, case_id=case_id, file_path=str(target), width=width, height=height)
    case.status = "imaging_uploaded"
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


@router.get("/images/{image_id}/file")
def get_image_file(image_id: UUID, db: Session = Depends(get_db)):
    image = db.get(XrayImage, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image.file_path)


@router.post("/cases/{case_id}/xray/{image_id}/analyze", response_model=list[XrayFindingRead])
def analyze_xray(case_id: UUID, image_id: UUID, db: Session = Depends(get_db)):
    image = db.get(XrayImage, image_id)
    case = db.get(ClinicalCase, case_id)
    if image is None or case is None or image.case_id != case_id:
        raise HTTPException(status_code=404, detail="Image or case not found")

    result = VisionService().analyze(image.file_path, case_id=case_id, image_id=image_id)
    db.execute(delete(XrayFinding).where(XrayFinding.image_id == image_id))
    findings: list[XrayFinding] = []
    for item in result["findings"]:
        finding = XrayFinding(
            image_id=image_id,
            case_id=case_id,
            category=item["category"],
            confidence=item["confidence"],
            tooth_number=item.get("tooth_number"),
            bbox=item["bbox"],
            polygon=item.get("polygon"),
            review_status="pending_review",
        )
        db.add(finding)
        findings.append(finding)
    case.status = "ready_for_review"
    db.commit()
    for finding in findings:
        db.refresh(finding)
    return findings


@router.get("/cases/{case_id}/findings", response_model=list[XrayFindingRead])
def list_findings(case_id: UUID, db: Session = Depends(get_db)):
    return db.execute(select(XrayFinding).where(XrayFinding.case_id == case_id)).scalars().all()


@router.patch("/findings/{finding_id}", response_model=XrayFindingRead)
def review_finding(finding_id: UUID, req: FindingReviewUpdate, db: Session = Depends(get_db)):
    finding = db.get(XrayFinding, finding_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found")
    finding.review_status = req.review_status
    if req.tooth_number is not None:
        finding.tooth_number = req.tooth_number
    db.commit()
    db.refresh(finding)
    return finding
