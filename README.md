# DentalAI-Copilot

An educational, single-clinic workbench with account registration/login, patient
appointments and an approval-aware Agent. Clinic records are shared; conversations
and approvals belong to their signed-in owner. No email or phone verification is used.

## Run and try the workflow

1. Copy `backend/.env.example` to `backend/.env` and configure the database/model.
2. Run `docker-compose up --build`, then open `http://localhost:4200`.
3. Register an account, then create an appointment in the form or connect the model.
4. Ask the Agent to create a synthetic patient. Review its proposed fields and explicitly approve.
5. Refresh/select that patient, start a new conversation, and ask to edit it, add/delete
   a note, or create/update/delete an appointment with an explicit date and timezone.
6. Check the execution trace, approval card and refreshed patient/appointment lists.
   Existing conversations can be reopened to continue their stored state after a refresh.

Mock mode preserves conversations but does not execute natural-language mutations.
Use a tool-capable provider model for the complete Agent workflow. The connection
button checks local configuration, not remote provider availability.

Accounts use HttpOnly cookies. Use the same hostname for frontend and API (the
default frontend API URL is `http://localhost:8000/api`). HTTPS deployments must
enable `COOKIE_SECURE=true` and configure `ALLOWED_ORIGINS` in backend settings.
Public registration grants access to the shared clinic; use synthetic data in this demo.

## Validate locally

```powershell
cd backend
.venv/Scripts/python.exe -m pip install -r requirements.txt
# A development/test PostgreSQL database with pgvector and CREATE SCHEMA permission:
$env:TEST_DATABASE_URL = 'postgresql+psycopg://dentalai:dentalai@localhost:5438/dentalai'
.venv/Scripts/python.exe -m pytest -q
cd ../frontend
npm run build
```

Integration tests create and remove only their own random schema. Without
`TEST_DATABASE_URL`, they are skipped. See [architecture](docs/architecture.md)
for the actual memory, approval, transaction and privacy boundaries.

## OpenRouter free-model setup

The chat service uses OpenRouter's free-model router by default. Create a local
`backend/.env` from `backend/.env.example`, then set the following values:

```env
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openrouter/free
OPENROUTER_API_KEY=your_openrouter_api_key
MOCK_LLM=false
```

`OPENROUTER_API_KEY` is ignored by Git; never put it in frontend code. Restart
the backend after changing this file, then use **Connect model** in the web UI.

## Semantic retrieval setup

To start over, open **数据导入 → 清空业务数据** and confirm the scope. This clears
all clinic business data and all members' conversations/checkpoints, while preserving
accounts, login sessions, backend configuration and the source DOX database. Active
imports/Agents block reset until they finish. The empty vector tables are prepared
for the configured embedding dimension, so a successful reset replaces the need
to migrate old vectors. This action cannot be undone.

RAG now uses LangChain's `RecursiveCharacterTextSplitter` and `OpenAIEmbeddings`
with the OpenRouter model `liquid/lfm-2.5-embedding-350m:free`, returning 1024-dimensional
float vectors. SQLAlchemy/pgvector still own storage, patient filtering and transactions.
Set `OPENROUTER_API_KEY` (or a separate `EMBEDDING_API_KEY`) in `backend/.env`:

```env
EMBEDDING_PROVIDER=openrouter
EMBEDDING_MODEL=liquid/lfm-2.5-embedding-350m:free
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_DIM=1024
RAG_CHUNK_SIZE=480
RAG_CHUNK_OVERLAP=72
```

**Existing installations:** stop API writes, update the configuration, install the
requirements, then preview and explicitly rebuild the index before restarting the API:

```powershell
cd backend
.venv/Scripts/python.exe -m app.maintenance.rebuild_embeddings
.venv/Scripts/python.exe -m app.maintenance.rebuild_embeddings --apply
```

The preview only prints counts. `--apply` sends existing indexed text and clinical notes
to the configured embedding provider, then atomically replaces the two vector indexes
and their column dimensions. Patient records remain intact. Do not use real patient
data with this free endpoint: [its model page](https://openrouter.ai/liquid/lfm-2.5-embedding-350m:free)
states that requests and embeddings may be retained for model training.

For Docker, stop the API first and run the same commands with
`docker-compose run --rm --no-deps api python -m app.maintenance.rebuild_embeddings`
(append `--apply` for the actual rebuild), after `docker-compose build api`.
Restart with `docker-compose up -d api`. Do not delete the PostgreSQL volume.

`MOCK_LLM` only controls chat. For completely offline teaching set
`EMBEDDING_PROVIDER=hash` and `MOCK_LLM=true`; changing vector mode requires a rebuild.
There is no silent fallback to hash if a real model request fails.

Run the synthetic comparison, which never reads patient data:

```powershell
cd backend
.venv/Scripts/python.exe -m app.maintenance.evaluate_embeddings --live
```

See [RAG implementation and migration](docs/rag.md) for the call chain, limits and tests.
