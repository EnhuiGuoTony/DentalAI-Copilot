# DentalAI Copilot

## Project map

- `frontend/`: Angular 22 workbench. Keep API types in `src/app/core/models/api.models.ts` and HTTP calls in `src/app/core/api/dentalai-api.service.ts`.
- `backend/`: FastAPI API. Routers live in `app/api/`, request/response models in `app/schemas/`, and business logic in `app/services/`.
- `backend/app/db/`: SQLAlchemy models and database setup; `backend/db-init/` initializes PostgreSQL + pgvector.
- `docs/architecture.md`: intended system design and safety boundaries.

## Local development

- Start the stack: `docker-compose up --build` (web: `http://localhost:4200`, API: `http://localhost:8000`).
- Frontend: from `frontend/`, run `npm install`, then `npm start`; validate with `npm run build`.
- Backend: from `backend/`, install `requirements.txt` and run `pytest`. Start locally with `uvicorn app.main:app --reload`.
- API routes are mounted below `/api`; use `/health` for a basic service check.

## Configuration and safety

- Create `backend/.env` from `backend/.env.example`. Never commit API keys, database credentials, patient data, or files under `backend/uploads/`.
- The frontend must not contain LLM provider keys; LLM configuration belongs in the backend environment.
- This is a clinical-workflow demo, not a diagnostic system. Preserve structured findings, RAG evidence, and tool traces; do not present model output as validated clinical advice.
- Preserve patient scoping when changing retrieval, imports, or API flows. Avoid logging sensitive patient content.

## Change conventions

- Keep frontend API models and backend schemas in sync when an endpoint contract changes.
- Prefer typed Pydantic/TypeScript models over unstructured dictionaries or `any`.
- Add or update focused pytest coverage for backend behavior changes. Run the narrow relevant test first, then broader checks when practical.
