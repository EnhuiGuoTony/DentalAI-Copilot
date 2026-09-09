# Resume Notes

## Project Title

DentalAI Copilot - Multimodal RAG Agent for Dental X-ray Analysis

## Resume Bullets

- Built a full-stack multimodal AI copilot using Angular, FastAPI, PostgreSQL/pgvector, OpenCV, and LLM tool orchestration.
- Implemented a custom RAG pipeline with clinical note chunking, deterministic demo embeddings, patient-scoped pgvector search, and citation-backed responses.
- Designed a LangGraph Agent workflow with explicit tools for X-ray finding retrieval, patient history search, and structured clinical summary generation.
- Integrated LangChain Gemini/OpenAI structured output with a mock fallback so the system runs before an API key is configured.
- Built a DOX MySQL ingestion layer that deidentifies legacy dental SaaS data and routes notes/templates/guidelines into pgvector while preserving treatments, diagnoses, perio findings, and attachments as structured Agent facts.
- Developed an Angular clinical workbench with X-ray upload, canvas-based overlays, confidence filtering, RAG evidence display, and Agent tool trace inspection.
- Structured AI output with JSON schemas, confidence scores, evidence references, and doctor-in-the-loop limitations.

## 30-Second Pitch

DentalAI Copilot is a full-stack multimodal RAG Agent project. A dentist uploads a dental X-ray, the system generates structured image findings, retrieves patient history through custom pgvector search, and uses a LangGraph Agent workflow with LangChain structured output to generate an evidence-backed clinical summary. The project demonstrates practical AI engineering: RAG, tool orchestration, multimodal context, structured outputs, and frontend/backend product integration.
