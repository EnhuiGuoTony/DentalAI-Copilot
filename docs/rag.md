# Implemented RAG pipeline

## Data flow

`records_service.apply_change(AddNote)` writes the note and calls
`VectorStore.add_document`. The common `chunk_text` function uses LangChain's
`RecursiveCharacterTextSplitter`. `EmbeddingService.embed_documents` calls
LangChain `OpenAIEmbeddings` through OpenRouter. After response validation,
SQLAlchemy inserts the chunks into PostgreSQL/pgvector in the same transaction
as the note (and the mutation receipt for an approved Agent write).

DOX imports also use the common splitter and batch embedding interface. Each
source replaces its old chunks within the import transaction; shortening a source
does not leave extra chunks behind. Manual patient re-indexing retains the patient
lock and replaces the note chunks, so repeated calls do not accumulate duplicates.

`PmsAgentService.search_evidence` calls `VectorStore.search` for the conversation's
selected patients and `search_knowledge` for shared knowledge. The query is encoded
once per VectorStore instance and reused across these searches. PostgreSQL computes
cosine distance and orders results; the score is `max(0, 1 - distance)`, not a
probability. Patient filtering remains in SQL. Evidence and source IDs enter the
Agent as tool results; a real chat model can synthesize them into its validated answer.

The separate case `/rag/query` API still returns a template summary. Its limitations
field makes that explicit. Semantic retrieval does not imply that this endpoint
now performs LLM generation. There is no reranker, relevance threshold, hybrid
search or HNSW/IVFFlat index in this change.

## Model and chunk limits

The requested model is `liquid/lfm-2.5-embedding-350m:free`, with 1024 dimensions
and a 512-token input window. Python sends `encoding_format="float"`, corresponding
to the JavaScript SDK's `encodingFormat: "float"`. `check_embedding_ctx_length=False`
keeps raw strings on the wire instead of using an incompatible OpenAI tokenizer.

Splitting tries paragraph, newline, Chinese/English punctuation and word boundaries,
then individual characters. The default limit is **480 UTF-8 bytes** with target
overlap **72 bytes**. This is a conservative byte budget, not an exact model token
count. It keeps Chinese and emoji from becoming unbounded inputs without downloading
model weights or a tokenizer. Natural boundaries can make actual overlap smaller.
Changing these settings changes the index fingerprint and requires rebuilding.

Long queries are split without overlap; their vectors are averaged with byte-length
weights and normalized. This preserves the tail of a query but is a simplifying
approximation and can dilute multi-topic questions. Documents remain separate chunks.

The embedding API supports configurable batching (default 16), timeout (30 seconds)
and retries (2). The service validates count, dimensions, finite values and nonzero
norms, then normalizes vectors. It never pads or truncates malformed vectors and
never falls back to the hash encoder on an API failure. Errors returned to API/Agent
omit provider response bodies, credentials and input text. API errors use 503 for
embedding failures and 409 when the index requires rebuilding.

`EMBEDDING_API_KEY` overrides `OPENROUTER_API_KEY`. Chat model keys/settings and
`MOCK_LLM` do not select the embedding provider. Explicit `EMBEDDING_PROVIDER=hash`
retains the old signed SHA-256 word projection for offline teaching; it is not a
trained semantic model. Tests select hash mode explicitly.

## Index compatibility and migration

If the old data is no longer needed, the frontend's **数据导入 → 清空业务数据**
action calls the authenticated `/api/workspace/reset` endpoint after confirmation.
It atomically truncates the explicitly listed clinic business and checkpoint tables
and changes the empty vector columns to the configured dimension. Accounts, login
sessions, checkpoint migration versions, backend settings, the source DOX database
and files on disk remain. All clinic members currently have this maintenance action;
the demo has no administrator role. The action is never registered as an Agent tool.
Ordinary API requests hold a shared workspace lock, and the Agent holds it throughout
SSE execution. Reset takes the exclusive lock and returns 409 when work is active.
No embedding calls occur during reset. A failed transaction retains all old data.

Each chunk stores an embedding fingerprint, provider, model, dimension and chunk
strategy in metadata. The fingerprint covers the endpoint/model (or hash algorithm),
dimension and splitter settings, excluding API credentials. Before retrieval or
insertion, both vector tables are checked for the actual PostgreSQL column dimension
and compatible metadata. Even equal-dimensional vectors cannot cross models. Checks
are reused only inside the same database transaction.

`python -m app.maintenance.rebuild_embeddings` is a read-only count preview.
`--apply` rebuilds both tables in one transaction. Run it during a maintenance window
with API writes stopped. It acquires a schema-scoped exclusive advisory lock and
table locks. Normal index operations hold a shared advisory lock for their transaction.
Existing notes are split from their full original content; knowledge or other sources
without a separate original-text table are split from their saved chunks. Old overlap
is not guessed or reconstructed, so those sources retain legacy boundaries. Re-import
their originals when exact full-document re-chunking is needed.

The script finishes all embedding calls before deleting old vectors. It then changes
column dimensions, inserts new vectors and commits. API failures, DDL failures and
insert failures roll back the entire migration. Patient records, clinical notes,
appointments, facts and conversations are not deleted. A second successful run
keeps stable chunk counts and IDs. Source IDs remain traceable.

This educational migration holds source/chunk metadata in memory, stages vectors
in transaction-local PostgreSQL temporary tables in batches, and holds a transaction
during remote calls. A failed run rolls back and starts again; it is not resumable.
Large deployments need a persistent shadow index and a controlled cutover.
Compatibility checks also scan chunk metadata once per transaction; they are intended
for the demo's data volume, not a substitute for a production index catalog.

## Evaluation and validation

`python -m app.maintenance.evaluate_embeddings --live` compares signed-hash vectors
and the configured model on six synthetic English/Chinese queries. It prints top-1
accuracy, mean reciprocal rank and each expected document's rank. It does not read
the database. This tiny smoke evaluation is not a clinical benchmark, and it does
not prove recall on imported records.

A live run on 2026-09-19 returned top-1 accuracy 2/6 and MRR 0.5333 for the hash
baseline, versus top-1 6/6 and MRR 1.0 for the requested Liquid model. This only
describes the six examples in the script; no claim is made about other datasets.

`tests/test_embedding.py` checks actual SDK HTTP payloads using MockTransport,
batching, Chinese/emoji splitting, long queries, malformed vectors and failures.
`tests/test_rag_integration.py` uses the existing random-schema PostgreSQL fixture
to cover old-dimension migration, model incompatibility, rollback, repeated rebuilds,
patient scoping, shortened imports, note rollback and concurrent rebuild protection.
The existing workspace tests still cover ownership, approvals, replay and cleanup.

## Provider data boundary

The [OpenRouter model page](https://openrouter.ai/liquid/lfm-2.5-embedding-350m:free)
states that requests and embeddings may be retained and used to train Liquid models.
Use synthetic data with this demo configuration. Embedding happens before the Agent's
identity substitution and PIIMiddleware; these do not anonymize embedding requests.
DOX's existing de-identification is also not comprehensive medical de-identification.
No automatic startup migration sends old patient data to the provider.
