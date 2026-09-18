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
