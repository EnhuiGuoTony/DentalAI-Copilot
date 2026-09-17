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
