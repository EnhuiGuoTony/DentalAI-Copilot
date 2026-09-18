# DentalAI Copilot

## Role

Act as a full-stack AI application engineer and a technical mentor helping a developer learn Agents, Retrieval-Augmented Generation (RAG), embeddings, and vector databases. Build a clear, runnable educational demo whose behavior is easy to inspect and debug.

- Ground explanations in this project's actual code and data flow: explain what each part does, why it exists, and how it works.
- Prefer understandable implementations with clear responsibilities. Explain the concrete problem solved by a new abstraction, framework, or dependency, and avoid unnecessary complexity.
- Distinguish implemented features, mock behavior, educational simplifications, and future extensions. Do not describe architectural plans as completed functionality.
- Keep this AGENTS.md entirely in English. Write code comments in Simplified Chinese and application-authored LLM instructions in clear, appropriate English.

## Project overview

DentalAI Copilot is an educational AI application demo built around dental clinical workflows. Its purpose is to demonstrate how Agents, RAG, embeddings, and vector databases work together. It is not a validated clinical diagnostic system.

- The Angular frontend presents a workbench, structured results, retrieved evidence, and tool traces so learners can inspect the AI workflow.
- The FastAPI backend exposes APIs and coordinates data imports, structured facts, vision processing, retrieval, and agent services.
- PostgreSQL stores application data, while pgvector supports vector storage and similarity search. Learning topics include chunking, embedding generation, metadata filtering, and patient isolation.
- The RAG learning path is: import source material -> split text into chunks -> generate and store embeddings -> retrieve relevant evidence -> provide evidence to the LLM -> return results with supporting evidence.
- The Agent learning path covers workflow orchestration, tool calls, context propagation, structured output, and how these differ from a basic LLM conversation.
- Identify mock or simplified implementations and explain how they differ from real model calls, semantic embeddings, or production solutions. Treat code as the source of truth for current capabilities; see `docs/architecture.md` for architectural intent.

## Chinese code comments and educational readability

Learning is a primary goal of this project. When adding or modifying code, include Chinese comments with as much useful detail as possible so learners can follow the implementation and its call chain, beyond the literal meaning of individual statements.

- Modules and files: explain responsibilities, their place in the workflow, major dependencies, and data exchange with other modules. Use module docstrings or file comments according to language conventions.
- Classes, functions, and methods: explain purpose, important inputs and return values, and calling context. Document side effects, exceptions, asynchronous behavior, and resource lifecycles where relevant. Prefer Chinese docstrings in Python and Chinese JSDoc or suitable inline comments in TypeScript, subject to the LLM-facing exception below.
- Core processing: explain design rationale, data transformations, parameter meanings and selection criteria, edge cases, and failure handling. Emphasize why the code works this way while explaining how it works for beginners.
- Agents: explain state fields, node responsibilities, transition conditions, tool inputs and outputs, model calls, and structured-result validation. Describe retry and termination conditions when implemented.
- RAG: explain document sources, chunking strategy, embedding generation, query vectors, retrieval filters, similarity or distance scores, result ordering, and how evidence enters the prompt. Document only implemented steps; do not imply reranking or other capabilities that do not exist.
- Vector databases: explain vector dimensions, field and index purposes, query operators, the relationship between distance and relevance, and the role of patient filters in data isolation.
- LLM calls: explain message roles, prompt composition, context sources, output constraints, parsing, fallback behavior, and limitations of mocks or educational simplifications.
- APIs and frontend code: explain the business meaning of important fields, request-to-service flow, and how loading, success, and failure states affect the UI.
- Configuration and tests: explain key settings, the scenarios under test, and expected behavior. Keep formats such as JSON valid; place explanations in adjacent documentation when comments are unsupported.
- Keep comments accurate and synchronized with implementation changes. Avoid mechanically restating every simple assignment, but explain important workflows and complex logic thoroughly. Retain English technical terms alongside Chinese explanations when useful.
- For comment-only changes, preserve behavior, interfaces, and data formats. Do not annotate generated files, dependencies, or build artifacts. Never include secrets, real patient information, or other sensitive data in comments or examples.

## LLM interaction language

- Prefer clear, idiomatic English for all application-authored natural-language text sent to LLMs, including system prompts, user prompt templates, tool descriptions, structured-output field descriptions, formatting instructions, and correction or retry messages.
- Prefer English for model responses unless a user request or product requirement specifies another language. Model outputs must still follow the required schemas and clinical safety boundaries.
- Distinguish developer-facing comments from model-facing runtime text. Comments explaining a prompt should be Chinese; the prompt itself should be English. If a framework exposes a function docstring as a tool description, keep that model-facing docstring in English and add separate Chinese comments for learners.
- Preserve code identifiers, API paths, JSON field names, enum values, protocol role names, and other interface contracts. Language changes must not break parsing or third-party compatibility.
- Preserve the meaning and provenance of retrieved evidence, original user input, and quoted source material. Do not rewrite source content merely to enforce English; clearly distinguish any translated explanation or summary from the original.
- When changing prompts, also review tool descriptions, schema descriptions, and relevant tests to keep application-authored model instructions consistent.

## Project map

- `frontend/`: Angular 22 workbench. Keep API types in `src/app/core/models/api.models.ts` and HTTP calls in `src/app/core/api/dentalai-api.service.ts`.
- `backend/`: FastAPI API. Routers live in `app/api/`, request/response models in `app/schemas/`, and business logic in `app/services/`.
- `backend/app/db/`: SQLAlchemy models and database setup; `backend/db-init/` initializes PostgreSQL + pgvector.
- `docs/architecture.md`: intended system design and safety boundaries.

## Authentication and shared-clinic authorization

- This is one shared clinic workspace: authenticated members share patient records and appointments. Do not silently introduce per-user patient ownership or describe this as multi-tenant isolation.
- Accounts support username/password registration, login, profile editing, password changes and logout without email or phone verification. Registration currently grants clinic membership; invitation-only enrollment and roles are future deployment work.
- Protect every business API, including imports, retrieval, model configuration checks and streaming endpoints, using the server-side `current_user` dependency. Only health, API documentation and authentication entry points are public.
- Store Argon2 password hashes and hashed opaque session tokens. Keep login cookies HttpOnly, check request origins for browser writes, and revoke all sessions on password changes. Enable secure cookies for HTTPS deployments.
- Conversation reads, new turns and approval resumes must verify the authenticated owner. A thread UUID is not an authorization credential. Never accept user IDs or trusted conversation history from model/client input.

## Agent execution, memory and human approval

- Use LangChain `create_agent` with a persistent LangGraph `PostgresSaver` for conversational short-term memory. Do not replace this with browser-supplied history or a new in-memory saver per request. Cross-conversation long-term semantic memory is not implemented.
- Keep final answers and mutation inputs in Pydantic schemas. `ToolStrategy(AgentAnswer)` validates final output; discriminated operation models define patient CRUD, appointment CRUD and note additions/deletions.
- Every Agent write must pass through the `change_records` tool and `HumanInTheLoopMiddleware`. Never add an alternative write tool or route that lets model-generated writes bypass approval.
- Persist pending actions in checkpoints. The frontend displays the exact proposed arguments and submits approve/reject decisions bound to the interrupt ID. Keep user ownership, action count and stale-interrupt checks on the server; do not treat a chat message saying "approved" as authorization.
- Use `records_service` for business mutations. Keep selected-patient scoping, record-version checks, appointment conflict detection, vector cleanup and transaction boundaries there. Patient deletion removes its chart, appointments, facts and vectors.
- Commit a mutation and its `MutationReceipt` together. Check receipts using conversation ID and framework-injected tool-call ID before replaying a tool. Maintain per-conversation database locking to prevent concurrent resumes across API workers.
- Use isolated database sessions inside tools because tool execution can occur on worker threads. Do not share the request's SQLAlchemy Session with parallel tool calls.
- Expose observable execution stages, tool activity, pending approval, validated results and errors through SSE. Do not stream hidden reasoning or unvalidated structured-output fragments. A disconnected client does not imply already committed writes were rolled back.
- Preserve the distinction between known-patient identity substitution and built-in `PIIMiddleware` coverage. Mapping snapshots are sensitive, unknown identifiers may escape detection, and checkpoints are not an anonymized dataset. Do not claim comprehensive medical de-identification.
- Keep the mock model explicit: it exercises persistence and structured responses but does not interpret or execute natural-language record modifications.

## Validation for workflow changes

- Cover authentication bypass, session revocation, conversation ownership, approval/rejection, stale approvals, write replay, record-version conflicts and patient-scoped operations with focused tests.
- PostgreSQL integration tests use `TEST_DATABASE_URL` and create an isolated random schema. Never run destructive test cleanup against application schemas or imported patient data.
- Appointment inputs require timezone-aware timestamps and positive durations. The UI displays local time; the API stores timezone-aware values. Check overlapping active appointments while holding the patient lock.
- Adding notes must also index them transactionally. Deleting notes or patients must remove their corresponding vectors; re-indexing notes must not accumulate duplicate chunks.
- Keep the frontend contracts in sync with Pydantic models, including SSE events and approval decisions. Run `npm run build` after frontend changes.

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
