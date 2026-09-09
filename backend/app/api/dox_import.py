from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.integrations.dox_mysql_importer import DoxMySqlImporter
from app.schemas.dox_import import DoxImportRequest, DoxImportSummary, DoxPreviewResponse

router = APIRouter(prefix="/dox-import", tags=["dox-import"])


@router.get("/preview", response_model=DoxPreviewResponse)
def preview_dox_import(db: Session = Depends(get_db)):
    return DoxMySqlImporter(db).preview()


@router.post("/run", response_model=DoxImportSummary)
def run_dox_import(req: DoxImportRequest, db: Session = Depends(get_db)):
    return DoxMySqlImporter(db).import_data(req)
