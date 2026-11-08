# Helios — Solution Overview

Helios is a proof-of-concept **HR Intelligence Copilot** for a synthetic organization (Northstar Global, ~120 employees). It is designed as an **operations assistant**, not a FAQ chatbot: it classifies requests, enforces authorization in Python, retrieves enterprise records and policies, computes leave and analytics deterministically, and requires human confirmation before consequential writes.



---

## 1. Problem and proposed solution

### Business problems addressed

| Stakeholder | Problem |
| --- | --- |
| Employees | Cannot easily check whether a leave plan is feasible (balance, holidays, projects, approvals). |
| Managers | Discover overlapping leave and coverage risk too late; approval actions are inconsistent if chat and API diverge. |
| HR | Need evidence-backed absence trends without invented rates or unsupported misconduct claims. |
| Security / compliance | An LLM that can talk about HR data can also leak salaries or approve leave by accident. |

### Solution approach

Helios separates four responsibilities:

1. **What the user is asking** — rule-first intent classification; optional LLM only when rules do not match.
2. **Facts and calculations** — leave balances, business days, overlaps, absence rates, z-scores from PostgreSQL via deterministic services.
3. **Whether data or tools may run** — backend RBAC before protected retrieval or mutation.
4. **Whether a write happens** — employee/manager/HR confirmation; the execute path is unreachable without resume after LangGraph `interrupt`.

Optional OpenRouter is used for intent fallback and grounded answer polishing. **Business numbers are never invented by the model.** If OpenRouter is unavailable or a rewrite fails grounding checks, Helios keeps the deterministic briefing.

---

## 2. Implementation

### High-level architecture

```text
Next.js UI  →  FastAPI  →  JWT + RBAC  →  LangGraph orchestrator
                                              ├─ Deterministic tools (leave, team, analytics, actions)
                                              ├─ Hybrid policy RAG (BM25 + dense + RRF) over pgvector
                                              └─ Human confirmation interrupt → simulated HRIS write
```

| Layer | Technology |
| --- | --- |
| Frontend | Next.js, React, TypeScript, Tailwind |
| Backend | FastAPI, Pydantic, SQLAlchemy |
| Orchestration | LangGraph (nodes, conditional edges, checkpoint, `interrupt`) |
| Database | PostgreSQL + pgvector (Docker Compose) |
| Embeddings | Local `minishlab/potion-retrieval-32M` (512-d) by default |
| Chat LLM | Optional OpenRouter (`OPENROUTER_API_KEY`) |
| Auth | PBKDF2 passwords, JWT sessions, roles: `employee`, `manager`, `hr_admin` |

### LangGraph workflow (simplified)

1. `classify_intent` — rules + conversation state; LLM classify only as fallback.
2. `authorize` — security layer; deny before protected retrieval when required.
3. Domain nodes — employee context, leave analysis, team conflict, workforce analytics, policy retrieval, approval decision, etc.
4. `prepare_workflow` — builds a pending consequential action when eligible.
5. `compose_response` — deterministic narration; optional grounded LLM polish.
6. `human_gate` — LangGraph `interrupt` when confirmation is required.
7. `execute_workflow` / `cancel_workflow` — only after explicit confirm/decline.

Chat approvals (`Approve leave request N`) call the same `decide_approval` service as `POST /api/approvals/{id}/decide`.

### RAG / knowledge

Policy documents are ingested with metadata (version, effective dates, department/location, conflict group). Retrieval:

1. Filter expired / not-yet-effective / superseded documents and role visibility.
2. BM25 lexical ranking + dense similarity.
3. Reciprocal Rank Fusion.
4. Optional cross-encoder rerank (`RERANK_ENABLED`).
5. If two current documents conflict, disclose the conflict and do not pick a winner.
6. If embeddings fail, degrade to BM25 and say so.

### Core workflows implemented

- **Employee leave planning** — balance, date capture, holidays in-window, projects, approvals, duration changes, draft preview, confirmation gate; insufficient balance blocks invalid submits.
- **Leave submit** — prepare → `/api/assistant/confirm` → pending manager/HR.
- **Manager conflict analysis** — overlapping leave for direct reports, coverage/risk, project context.
- **Manager chat approve/reject** — authorized scope, rejection reasons, audit, DB status.
- **HR Admin analytics** — department/month/office comparisons, anomalies, evidence-backed recommendations.
- **Transfer / routing mismatch** — line manager vs approval manager can diverge (seeded Liam Chen scenario).
- **Security** — cross-employee salary denial, org-scope denial for employees, injection/secret refusal, REST 403 on unauthorized profiles.

### Repository layout (main pieces)

| Path | Role |
| --- | --- |
| `backend/app/ai/graph.py` | LangGraph orchestration |
| `backend/app/ai/context.py` | Intent, follow-ups, approval parsing |
| `backend/app/ai/retrieval.py` | Hybrid RAG |
| `backend/app/ai/narrate.py` | Deterministic response composition |
| `backend/app/services/*` | Leave engine, team ops, analytics, actions |
| `backend/app/knowledge/policies/` | Seeded policy corpus |
| `frontend/app/*` | HR workspace pages (assistant, leave, team, analytics, …) |
| `docker-compose.yml` | Postgres + backend + frontend |

Startup instructions live in `README.md`. Environment template: `.env.example`.

---

## 3. Technical decisions

| Decision | Rationale |
| --- | --- |
| Deterministic tools for HR math | Prevents hallucinated balances, rates, and headcount. |
| Rule-first intent classification | Stable demos and security-critical routing; LLM only fills gaps. |
| LangGraph with real `interrupt` | Human-in-the-loop is structural, not a prompt suggestion. |
| Same approval service for chat and REST | Avoids inconsistent authorization between channels. |
| Hybrid BM25 + dense + RRF | Robust to keyword and paraphrase policy questions. |
| Fixed `ORG_CLOCK` | Reproducible leave/Atlas/absence demos. |
| Synthetic HRIS (`simulated_hris`) | No external Workday/SAP required for the take-home POC. |
| Local embeddings by default | Runs without a paid embedding API; OpenRouter is optional for prose. |
| Grounded polish (`is_grounded`) | LLM may rephrase but cannot introduce new numbers or claims. |
| PostgreSQL + pgvector in Compose | Matches enterprise vector search; SQLite only as optional laptop mode. |

---

## 4. Assumptions

1. The audience is evaluating an **enterprise AI POC**, not a production multi-tenant HRIS.
2. Demo users use the seeded accounts and password documented in `README.md`.
3. Docker Desktop is available; Compose is the supported full-stack path.
4. Attendance, leave, projects, and policies in the database are the system of record for the demo.
5. OpenRouter credits may be limited; Helios remains usable without LLM via deterministic paths.
6. “Organization today” for all business logic is **2026-11-07** unless `ORG_CLOCK` is changed.
7. Managers only see their authorized reporting line; HR Admin has organization scope for analytics.
8. Leave approval can use `approval_manager_id`, which may differ from `manager_id` after transfers.

---

## 5. Limitations

1. **Synthetic data only** — no live HRIS, payroll, or identity provider integration.
2. **Intent routing** still relies heavily on rules; unusual paraphrases can misroute (improved for phrases like `plan leave` / `apply for leave`, but not exhaustive).
3. **LLM polish** often falls back to deterministic text when grounding rejects a rewrite — answers may feel “templated” even when OpenRouter works.
4. **LangGraph checkpoints** use a path suitable for a single API process; multi-replica resume needs a shared checkpointer.
5. **Scale** — analytics scan attendance rows in-process; not load-tested for 5,000+ employees.
6. **Holiday calendars** are seeded for the demo offices, not a live government feed.
7. **README** focuses on startup; this document carries architecture depth for assessment readers.
8. **Automated pytest suite** may be incomplete in a given delivery tree; live Docker demos are the primary verification path for this POC.
9. Demo DB state drifts after approve/reject exercises; `docker compose down -v` reseeds a clean world.

---

## 6. Potential future improvements

1. Replace regex-heavy intent with a constrained classifier + structured entity extraction, keeping security rules first.
2. Move LangGraph checkpoints to PostgreSQL for horizontal API scaling.
3. Materialize monthly attendance aggregates for large workforces.
4. Integrate real HRIS adapters behind the same authorization and tool interfaces.
5. Add SSO (OIDC/SAML), shorter JWT lifetimes, and secret management.
6. Background workers for reindex, bulk reports, and audit retention jobs.
7. Stronger monitoring: denied sensitive reads, tool latency, retrieval degraded mode alerts.
8. Official holiday feed ingestion and multi-country calendar governance.
9. CI retrieval evaluation on every policy corpus change.
10. Richer transfer-correction UX while keeping HR approval for reporting-line changes.

---

## 7. Stakeholder trade-offs (as implemented)

| Trade-off | Helios stance |
| --- | --- |
| Automation vs oversight | Automate analysis and draft prep; never silent approve/submit. |
| Experience vs security | Fast self-service answers; deny cross-employee private data before retrieval. |
| Cost vs capability | Works without LLM; optional OpenRouter for polish/classify only. |
| Maintainability vs integration depth | Clear service boundaries; simulated HRIS kept swappable later. |

---

## 8. How to evaluate quickly

1. Start with `README.md` (`docker compose up --build`).
2. Employee (`aisha.rahman@northstar.example`): leave balance → December plan → holidays/projects/approvals → one-day confirm.
3. Manager (`james.okonkwo@northstar.example`): overlapping leave → chat approve/reject.
4. HR (`helena.vogel@northstar.example`): department/month/office absence analytics.
5. Security: ask for another employee’s salary; confirm denial / API 403.

API reference: `http://localhost:8000/docs`.
