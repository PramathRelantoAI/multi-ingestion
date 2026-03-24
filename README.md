# Multi-Source Ingestion RAG Question filler

This document defines the end-to-end architecture for a LangChain/LangGraph multi-source ingestion and form-filling system.

---

## 1) Goals and Scope

- Automate answer extraction for structured questionnaires from multi-source enterprise artifacts.
- Keep ingestion and answer-generation decoupled so either can run independently.
- Support cloud-native operation with observable, retry-safe, and scalable components.
- Persist outputs with citations and conflict metadata for downstream review.
- Make section workers dynamic through registry/configuration (no hardcoded worker files required).

---

## 2) System Context

```mermaid
flowchart LR
    Sources[Drive/Zoom/Email/Other] --> Sync[Ingestion Connectors]
    Sync --> GCSRaw[(GCS raw/)]
    GCSRaw --> Proc[Preprocess + Normalize]
    Proc --> GCSProcessed[(GCS processed/)]
    GCSProcessed --> Dispatch[Dispatch Events]
    Dispatch --> Topics[Queue / PubSub]
    Topics --> Ingest[Ingestion Service]
    Ingest --> VS[Vector Search]
    Ingest --> OutTopic[OUTPUT_TOPIC]
    OutTopic --> Retriever[Retrieval Scheduler]
    Retriever --> AgentV2[LangGraph Supervisor + Dynamic Workers]
    AgentV2 --> PG[(PostgreSQL)]
    AgentV2 --> Form[(GCS form output JSON)]
    API[FastAPI: /api/runs] --> AgentV2
```

---

## 12) AgentV2 Architecture (Reference)

This section captures the detailed AgentV2 flow inside the main architecture document.

# AgentV2 Architecture

This document describes the architecture of **agentV2**: a LangGraph-based Supervisor + 6 Worker form-filling pipeline that extracts answers from retrieved RAG chunks and produces a structured form output (BRD §18) with citations and optional conflict tracking.

---

## 1. High-Level Overview

AgentV2 is triggered after RAG retrieval completes for an opportunity. It runs a **LangGraph StateGraph** that:

1. **Runs 6 workers in parallel** — each worker is a batch of SASE DoR questions; each batch gets its own LLM call with partitioned chunks.
2. **Validates** candidate answers (picklist, type, required).
3. **Detects conflicts** — worker-flagged conflicts, multiple differing values, or low confidence.
4. **Optionally recalls workers** — up to `MAX_RECALL_ROUNDS` (default 2) with recall context when conflicts exist.
5. **Selects final answers** per question (by confidence, then single agent), persists to PostgreSQL (answers + citations + conflicts).
6. **Builds form output** — BRD §18 JSON with dependency-filtered active questions; writes to GCS.

```mermaid
flowchart TB
    subgraph External["External Triggers"]
        RAG[RAG Ingestion Complete]
        API[API / Extract]
        Scheduler[Retrieval Scheduler]
    end

    subgraph Retrieval["Retrieval"]
        VS[Vertex AI Vector Search]
        Retrievals[(retrievals: q_id → chunks)]
    end

    subgraph AgentV2["agentV2 Pipeline"]
        Graph[LangGraph StateGraph]
        Workers[6 Workers]
        Supervisor[Supervisor]
        Store[(Store / PG)]
        FormOut[Form Output]
    end

    subgraph Outputs["Outputs"]
        PG[(PostgreSQL)]
        GCS[(GCS Form JSON)]
    end

    RAG --> Scheduler
    API --> Retrieval
    Scheduler --> Retrieval
    Retrieval --> VS
    VS --> Retrievals
    Retrievals --> Graph
    Graph --> Workers
    Graph --> Supervisor
    Supervisor --> Store
    Supervisor --> FormOut
    Store --> PG
    FormOut --> GCS
```

---

## 2. LangGraph State Flow

The pipeline is implemented as a **LangGraph `StateGraph`** with typed state (`AgentV2State`). Nodes update the state; conditional edges decide whether to recall workers or proceed to final selection.

```mermaid
flowchart LR
    START([START]) --> run_all_workers
    run_all_workers --> validate
    validate --> detect_conflicts
    detect_conflicts --> route{conflicts and\nrecall_round < max?}
    route -->|Yes| recall_workers
    route -->|No| select_final
    recall_workers --> run_all_workers
    select_final --> form_output
    form_output --> END([END])
```

**Nodes:**

| Node | Responsibility |
|------|----------------|
| **run_all_workers** | Partition retrievals by batch; run 6 workers in parallel via `run_all_workers()`; set `candidate_answers`, clear `recall_context`. |
| **validate** | Validate candidates (picklist, type, required); set `validation_errors`. |
| **detect_conflicts** | Detect conflicts (worker-flagged, multiple values, low confidence); set `conflicts`. |
| **recall_workers** | Build `recall_context` (reason + existing candidates per question); increment `recall_round`. |
| **select_final** | Select final answer per question; persist to PG (`persist_final_answers`, `persist_conflicts`); set `final_answers`. |
| **form_output** | Build BRD §18 form JSON (dependency-filtered); write to GCS; set `form_output`. |

**Routing:** After `detect_conflicts`, if there are conflicts and `recall_round < MAX_RECALL_ROUNDS`, the graph goes to `recall_workers` then back to `run_all_workers`. Otherwise it goes to `select_final` → `form_output` → END.

---

## 3. State Schema (AgentV2State)

State is a TypedDict; each node can emit a partial update.

```mermaid
erDiagram
    AgentV2State {
        string opportunity_id
        ChunksByQuestion retrievals
        list CandidateAnswer candidate_answers
        dict conflicts
        int recall_round
        dict recall_context
        dict final_answers
        dict form_output
        dict validation_errors
    }

    CandidateAnswer {
        string question_id
        string agent_id
        any candidate_answer
        float confidence
        list CandidateSource sources
        bool conflict
        string conflict_reason
        list conflict_details
        list answer_basis
    }

    CandidateSource {
        string source
        string chunk_id
        float retrieval_score
        string excerpt
        string source_type
    }

    AgentV2State ||--o{ CandidateAnswer : "candidate_answers"
    CandidateAnswer ||--o{ CandidateSource : "sources"
```

- **retrievals**: `q_id → list[RetrievedChunk]` (from retrieval pipeline).
- **candidate_answers**: One list from the current worker run (replaced each `run_all_workers`).
- **conflicts**: `question_id → reason` (e.g. `worker_flagged_conflict`, `multiple_differing_values`, `low_confidence`).
- **recall_context**: Built by supervisor; passed into workers on recall (reason + existing candidates per question).
- **final_answers**: `question_id → { answer, citations, confidence, status, ... }`.
- **form_output**: BRD §18 structure: `{ form_id, opportunity_id, answers: [{ question_id, answer }] }`.

---

## 4. Workers: Batches and LLM Flow

Workers are **batch-scoped**: each of the 6 batches (from `batch_registry` / `sase_batches`) has a fixed set of questions. Retrievals are **partitioned by batch** so each worker only sees chunks for its questions.

```mermaid
flowchart TB
    subgraph Input["Input"]
        Retrievals[retrievals: ChunksByQuestion]
    end

    subgraph Partition["Partition by batch"]
        P1[batch_order 1]
        P2[batch_order 2]
        P3[batch_order 3]
        P4[batch_order 4]
        P5[batch_order 5]
        P6[batch_order 6]
    end

    subgraph Workers["Workers (parallel)"]
        W1[Worker 1: Use Case]
        W2[Worker 2: Customer Tenant]
        W3[Worker 3: Infrastructure]
        W4[Worker 4: Mobile Users]
        W5[Worker 5: ZTNA]
        W6[Worker 6: Remote Network]
    end

    subgraph PerWorker["Per worker"]
        Chunks[batch_chunks]
        Prompt[System + User prompt]
        LLM[LLMClient: Gemini]
        Schema[Pydantic schema]
        Candidates[CandidateAnswer list]
    end

    Retrievals --> Partition
    Partition --> P1 & P2 & P3 & P4 & P5 & P6
    P1 --> W1
    P2 --> W2
    P3 --> W3
    P4 --> W4
    P5 --> W5
    P6 --> W6

    W1 & W2 & W3 & W4 & W5 & W6 --> Chunks
    Chunks --> Prompt
    Prompt --> LLM
    Schema --> LLM
    LLM --> Candidates
```

**Batch → Agent ID mapping** (`config.BATCH_ID_TO_AGENT_ID`):

| batch_id | agent_id |
|----------|----------|
| sase_use_case_details | agent_use_case |
| sase_customer_tenant | agent_customer_tenant |
| sase_infrastructure_details | agent_infrastructure |
| sase_mobile_user_details | agent_mobile_users |
| sase_ztna_details | agent_ztna |
| sase_remote_network_svc_conn | agent_remote_network |

Each worker uses:

- **batch_registry** + **field_loader**: batch definition, field definitions, dynamic Pydantic schema.
- **prompt_builder**: system prompt (section + per-question prompts), user prompt with context text.
- **LLMClient**: Vertex AI Gemini; optional CachedContent for system prompt.
- **recall_context**: When present, prepended to user context so the LLM can re-evaluate with conflict reason and existing candidates.

Output of `run_all_workers` is a **flat list of `CandidateAnswer`** (one per question per batch), merged from all 6 workers.

---

## 5. Supervisor Logic

The supervisor (no separate “agent”; pure functions in `agentV2/supervisor/logic.py`) handles validation, conflict detection, recall context, and final selection.

```mermaid
flowchart LR
    subgraph Validate["validate_candidates"]
        V1[Group by question_id]
        V2[Field def per question]
        V3[Picklist / type / required]
        V4[validation_errors]
    end

    subgraph Detect["detect_conflicts"]
        D1[Group by question_id]
        D2[Worker-flagged conflict?]
        D3[Multiple differing values?]
        D4[Low confidence?]
        D5[conflicts dict]
    end

    subgraph Recall["build_recall_context"]
        R1[conflicts + candidate_answers]
        R2[reason string]
        R3[existing_by_question]
    end

    subgraph Select["select_final_answers"]
        S1[Group by question_id]
        S2[Filter valid: no conflict, has value]
        S3[Best by confidence, tie-break agent_id]
        S4[final_answers + status]
    end

    candidate_answers --> Validate
    candidate_answers --> Detect
    conflicts --> Recall
    candidate_answers --> Recall
    candidate_answers --> Select
```

- **validate_candidates**: Uses `field_loader` + `batch_registry` for field definitions; checks picklist/type/required; returns `validation_errors`.
- **detect_conflicts**: Flags questions with worker-flagged conflict, multiple different non-null answers, or max confidence below `LOW_CONFIDENCE_THRESHOLD` (0.5).
- **build_recall_context**: Builds `{ reason, existing_by_question }` for re-evaluation.
- **select_final_answers**: For each question, picks best valid candidate (confidence, then agent_id); sets `status` to `confirmed` (with evidence) or `needs_review`.

---

## 6. Dependency Engine and Form Output

Before building the final form JSON, **active questions** are determined by the **dependency engine** (`agentV2/dependency`): rules like `depends_on_question` + `depends_on_condition` (e.g. “Yes”, or value in list). Only active questions are included in the form output.

```mermaid
flowchart LR
    final_answers --> get_active_question_ids
    dependency_rules[(dependency_rules.json)] --> get_active_question_ids
    get_active_question_ids --> active_set[active question IDs]
    active_set --> build_form_output
    build_form_output --> form_json[form_id, answers[]]
    form_json --> write_form_output_gcs
    write_form_output_gcs --> GCS[(GCS)]
```

- **build_form_output**: Builds `{ form_id, opportunity_id, answers: [{ question_id, answer }] }`; only includes active questions and existing final answers.
- **write_form_output_gcs**: Writes JSON to GCS (e.g. `{opportunity_id}/responses/form_output_*.json`).

---

## 7. Persistence (Store)

When PostgreSQL is configured (Cloud SQL or PG_* env), the **store** persists:

1. **Final answers** — `persist_final_answers(opportunity_id, final_answers)`: upserts into `answers` (typed columns, status, confidence, source_count) and inserts **citations** (BRD §13).
2. **Conflicts** — `persist_conflicts(opportunity_id, conflicts, candidate_answers, final_answers)`: inserts into `conflicts` and conflict-related **citations** (BRD §15).

```mermaid
flowchart TB
    select_final_node --> persist_final_answers
    select_final_node --> persist_conflicts

    persist_final_answers --> answers_table[(answers)]
    persist_final_answers --> citations_table[(citations)]

    persist_conflicts --> conflicts_table[(conflicts)]
    persist_conflicts --> citations_table
```

If no DB is configured, both functions are no-ops.

---

## 8. Component Map

```mermaid
flowchart TB
    subgraph Entry["Entry points"]
        graph_run["agentV2.graph.run()"]
        retrieval_scheduler["retrieval_scheduler (Cloud Function)"]
        run_agentv2_local["scripts/run_agentv2_local.py"]
    end

    subgraph agentV2["agentV2"]
        graph["graph.py"]
        state["state.py"]
        config["config.py"]

        subgraph workers["workers"]
            runner["runner.py"]
        end

        subgraph supervisor["supervisor"]
            logic["logic.py"]
        end

        subgraph form_output["form_output"]
            generator["generator.py"]
        end

        subgraph dependency["dependency"]
            engine["engine.py"]
        end

        subgraph store["store"]
            persistence["persistence.py"]
        end
    end

    subgraph shared["Shared services"]
        batch_registry["batch_registry"]
        field_loader["field_loader"]
        prompt_builder["prompt_builder"]
        llm_client["LLMClient"]
    end

    graph_run --> graph
    retrieval_scheduler --> graph_run
    run_agentv2_local --> graph_run

    graph --> state
    graph --> workers
    graph --> supervisor
    graph --> form_output
    graph --> store

    workers --> runner
    runner --> batch_registry
    runner --> field_loader
    runner --> prompt_builder
    runner --> llm_client

    supervisor --> logic
    logic --> batch_registry
    logic --> field_loader

    form_output --> generator
    generator --> dependency
    store --> persistence
```

---

## 9. End-to-End Data Flow (Summary)

```mermaid
sequenceDiagram
    participant Trigger
    participant Retrieval
    participant Graph
    participant Workers
    participant Supervisor
    participant Store
    participant GCS

    Trigger->>Retrieval: opportunity_id
    Retrieval->>Retrieval: Vector search per question
    Retrieval->>Graph: opportunity_id, retrievals

    Graph->>Workers: partition_by_batch(retrievals)
    Workers->>Workers: run_all_workers (6 in parallel)
    Workers->>Graph: candidate_answers

    Graph->>Supervisor: validate_candidates
    Supervisor->>Graph: validation_errors

    Graph->>Supervisor: detect_conflicts
    Supervisor->>Graph: conflicts

    alt conflicts and recall_round < max
        Graph->>Supervisor: build_recall_context
        Graph->>Workers: run_all_workers(recall_context)
    else no conflicts or max rounds
        Graph->>Supervisor: select_final_answers
        Supervisor->>Graph: final_answers
        Graph->>Store: persist_final_answers, persist_conflicts
        Graph->>Graph: build_form_output
        Graph->>GCS: write_form_output_gcs
    end
```

---

## 10. Key Files Reference

| Area | Path |
|------|------|
| Graph & entry | `agentV2/graph.py` — `build_graph()`, `get_graph()`, `run()` |
| State | `agentV2/state.py` — `AgentV2State`, `CandidateAnswer`, `CandidateSource` |
| Config | `agentV2/config.py` — batch→agent map, thresholds, form ID prefix |
| Workers | `agentV2/workers/runner.py` — `partition_by_batch`, `run_worker_batch`, `run_all_workers` |
| Supervisor | `agentV2/supervisor/logic.py` — validate, detect_conflicts, build_recall_context, select_final_answers |
| Form output | `agentV2/form_output/generator.py` — build_form_output, write_form_output_gcs |
| Dependency | `agentV2/dependency/engine.py` — get_active_question_ids |
| Store | `agentV2/store/persistence.py` — persist_final_answers, persist_conflicts |
| Batches & fields | `src/services/agent/batch_registry.py`, `field_loader.py`, `prompt_builder.py` |
| LLM | `src/services/llm/client.py` — LLMClient |
| Trigger (GCP) | `functions/retrieval_scheduler.py` — retrieval + agentV2.run() |

This architecture keeps a single, linear LangGraph flow with one optional recall loop, shared state, and clear separation between workers (batch-level LLM extraction), supervisor (validation and selection), and output (form JSON + persistence).

---

## 3) Logical Architecture

### A. Ingestion Layer (Dynamic Sources)

**Primary responsibility:** normalize source artifacts and index them for retrieval.

- Connector registry pattern (`connector_type -> connector_impl`).
- Source configs stored per tenant/org (enabled flag, auth ref, filters, schedule).
- Pipeline stages:
  - discover changes
  - fetch content
  - normalize to common schema
  - dedupe
  - chunk by strategy
  - embed + upsert to vector index

**Connector contract:**

```python
class BaseConnector(Protocol):
    connector_type: str
    def discover(self, cursor: dict | None) -> list[SourceItem]: ...
    def fetch(self, item: SourceItem) -> RawDocument: ...
    def to_ingestion_records(self, raw_doc: RawDocument) -> list[IngestionRecord]: ...
```

### B. Retrieval + Orchestration Layer

**Primary responsibility:** retrieve context and produce final answers.

- Retrieval scheduler:
  - triggered by ingestion completion
  - retrieves chunks per question
  - sends `ChunksByQuestion` payload to LangGraph
- Agent graph:
  - loads active worker sections from section registry
  - runs workers in parallel per section
  - validates candidates, detects conflicts, optional recall loop
  - selects final answers and persists outputs

### C. Serving/API Layer

- API supports:
  - direct run trigger
  - run status/results
  - answer override
  - section/prompt management

### D. Persistence and Output Layer

- PostgreSQL:
  - answers, citations, conflicts, run metadata
  - source config and section registry metadata
- GCS:
  - raw/processed assets and final form output JSON

---

## 4) Runtime Data Flow

### Flow 1: Scheduled Ingestion to Answer Generation

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant I as Ingestion Connectors
    participant N as Normalizer/Chunker
    participant V as Vector Search
    participant R as Retrieval Scheduler
    participant A as LangGraph Agent
    participant D as PostgreSQL/GCS

    S->>I: run enabled connectors
    I->>N: normalized records
    N->>V: chunk/embed/upsert
    N->>R: completion event
    R->>V: retrieve context by question
    R->>A: run(payload)
    A->>D: persist answers/citations/conflicts/form
```

### Flow 2: API-Driven Answer Generation

- Client submits run request with tenant/opportunity (+ optional question scope).
- Retrieval + agent pipeline executes.
- API returns run output and status.

---

## 5) Dynamic Worker Model (Required)

Workers are **data-driven section workers**, not static code-defined workers.

### Section registry schema (example)

```json
{
  "section_id": "sase_customer_tenant",
  "worker_id": "agent_customer_tenant",
  "active": true,
  "batch_order": 2,
  "question_ids": ["DOR-010", "DOR-011"],
  "system_prompt_template": "You are an expert in {{section_name}}...",
  "user_prompt_template": "Use only provided chunks and cite evidence.",
  "model": "gemini-2.5-flash"
}
```

### Worker execution behavior

1. Load active sections from registry ordered by `batch_order`.
2. Partition retrieval chunks by section `question_ids`.
3. Build per-section schema from field definitions.
4. Build prompts dynamically (system + user).
5. Execute all section workers in parallel.
6. Merge to single `candidate_answers` list.

### Graph nodes

- `load_sections`
- `run_all_workers_dynamic`
- `validate_candidates`
- `detect_conflicts`
- `recall_workers` (conditional)
- `select_final_answers`
- `build_form_output`
- `persist_outputs`

---

## 6) Data Architecture

### Storage tiers

- **GCS raw/**: source-of-truth copies.
- **GCS processed/**: normalized text assets.
- **Vector index(es)**: semantic retrieval substrate.
- **PostgreSQL**: answers, citations, conflicts, run and config metadata.

### Canonical object path

`gs://<bucket>/<tenant_id>/<opportunity_id>/<tier>/<source>/<object_name>`

### Core payloads

- `RetrievedChunk`: chunk id, excerpt, source metadata, retrieval score.
- `ChunksByQuestion`: question id -> list[RetrievedChunk].
- `CandidateAnswer`: worker-proposed value + confidence + sources + conflict flags.
- `FinalAnswer`: selected answer + citations + status.

---

## 7) Deployment Architecture

```mermaid
flowchart TB
    subgraph Compute
      ING[Ingestion service/jobs]
      RET[Retrieval scheduler]
      API[FastAPI]
      AG[LangGraph runtime]
    end
    subgraph Messaging
      MQ[PubSub/Queue]
    end
    subgraph Data
      GCS[(Blob Storage)]
      VDB[(Vector Search)]
      PG[(PostgreSQL)]
    end

    ING --> GCS
    ING --> VDB
    ING --> MQ
    MQ --> RET --> VDB
    RET --> AG
    API --> AG
    AG --> PG
    AG --> GCS
```

---

## 8) Reliability, Security, and Operations

### Reliability

- Event-driven decoupling with retries and dead-letter queue.
- Idempotency keys for ingestion and run execution.
- Conflict detection + optional recall improves answer quality.

### Security

- IAM/service-account access per component.
- Secrets in secret manager or gitignored secure env files.
- Least privilege for source connectors and data stores.

### Observability

- Structured logs with `run_id` and `tenant_id`.
- Stage-level metrics: ingest latency, retrieval hit rate, conflict rate, confidence distribution.
- Alerts on ingestion lag, failed runs, and abnormal conflict spikes.

---

## 9) Extensibility

- New source: add connector + optional preprocessing/chunker strategy.
- New question set: update section/field/prompt metadata only.
- New sink: add persistence adapter (CRM/export systems).

---

## 10) Primary Code Map (Target)

- Ingestion:
  - `ingestion/connectors/base.py`
  - `ingestion/connectors/registry.py`
  - `ingestion/pipeline/ingest_job.py`
- Agent:
  - `agent/graph.py`
  - `agent/workers/runner.py`
  - `agent/sections/section_registry.py`
  - `agent/supervisor/logic.py`
- API:
  - `ui/api/` (run and section management endpoints)

This architecture preserves the proven PANW flow while making worker orchestration fully dynamic through section/prompt metadata.

---

## 11) Project File Structure (Recommended)

```text
multi-ingestion-pipeline/
├── ARCHITECTURE.md
├── pyproject.toml
├── README.md
├── configs/
│   ├── settings.py
│   ├── .env.example
│   └── secrets/
│       └── .env.example
│
├── ingestion/
│   ├── connectors/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── drive_connector.py
│   │   ├── zoom_connector.py
│   │   └── email_connector.py
│   ├── preprocess/
│   │   ├── documents.py
│   │   ├── zoom_vtt.py
│   │   └── email_threads.py
│   ├── chunking/
│   │   ├── base.py
│   │   ├── documents_chunker.py
│   │   ├── zoom_chunker.py
│   │   └── email_chunker.py
│   ├── pipeline/
│   │   ├── ingest_job.py
│   │   ├── normalize.py
│   │   ├── dedupe.py
│   │   ├── embed_upsert.py
│   │   └── dispatch.py
│   ├── models/
│   │   ├── ingestion_record.py
│   │   └── source_config.py
│   └── store/
│       ├── gcs_store.py
│       ├── checkpoint_repo.py
│       └── source_config_repo.py
│
├── retrieval/
│   ├── scheduler.py
│   ├── vector_search.py
│   ├── reranker.py
│   └── questions_loader.py
│
├── agent/
│   ├── graph.py
│   ├── state.py
│   ├── config.py
│   ├── sections/
│   │   ├── section_registry.py
│   │   ├── prompt_templates_repo.py
│   │   └── field_definitions_repo.py
│   ├── workers/
│   │   ├── runner.py
│   │   ├── dynamic_worker.py
│   │   └── schema_builder.py
│   ├── supervisor/
│   │   ├── validate.py
│   │   ├── conflicts.py
│   │   ├── recall.py
│   │   └── select_final.py
│   ├── output/
│   │   ├── dependency_engine.py
│   │   └── form_generator.py
│   └── persistence/
│       ├── answers_repo.py
│       ├── citations_repo.py
│       └── conflicts_repo.py
│
├── api/
│   ├── main.py
│   ├── routes/
│   │   ├── runs.py
│   │   ├── connectors.py
│   │   └── sections.py
│   ├── schemas/
│   │   ├── run_models.py
│   │   ├── connector_models.py
│   │   └── section_models.py
│   └── deps/
│       ├── auth.py
│       └── db.py
│
├── ui/
│   └── web/  # Next.js frontend
│
├── functions/  # optional cloud entrypoints
│   ├── drive_sync.py
│   ├── gcs_file_processor.py
│   ├── pubsub_dispatch.py
│   ├── rag_ingestion.py
│   └── retrieval_scheduler.py
│
├── workflows/
│   ├── ingestion_pipeline.yaml
│   └── rag_answer_pipeline.yaml
│
├── scripts/
│   ├── run_local_ingestion.py
│   ├── run_local_agent.py
│   └── seed_section_registry.py
│
└── tests/
    ├── ingestion/
    ├── retrieval/
    ├── agent/
    └── api/
```
