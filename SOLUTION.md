# Helios — Solution Overview

Helios is a proof-of-concept **HR Intelligence Copilot** for a synthetic organization (Northstar Global). It is an **operations assistant**: it classifies the request, enforces authorization in Python, retrieves policies and enterprise records, computes leave and attendance in deterministic services, and writes only after a human confirms a durable pending action.

Business numbers come from those services. The optional language model may classify an unmatched question, choose among a fixed tool list, resolve a follow-up, or explain evidence. It does not run SQL, grant access, or invent balances, rates, or successful writes.

---

## 1. Problem and proposed solution

### Business problems addressed

| Stakeholder | Problem |
| --- | --- |
| Employees | Checking whether a leave plan is feasible (balance, holidays, projects, notice, approvals) takes several systems. |
| Managers | Overlapping leave and coverage risk show up late, and a chat approval must match the approvals API. |
| HR | Absence trends need evidence (department, month, office) without invented rates or unsupported misconduct claims. |
| Security / compliance | An assistant that can discuss HR data can also leak salaries, widen scope, or approve leave if the model is trusted to enforce policy. |

### Solution approach

Helios keeps five responsibilities in code:

1. **Whether the message is allowed** — a security assessment runs on the raw text before an intent is chosen. A harmful instruction mixed into an ordinary question denies the whole message.
2. **What the user is asking** — rules and a structured query spec first; the model classifies only when those do not match an allowed intent.
3. **Facts and calculations** — leave balances, business days, overlaps, absence rates, and z-scores from PostgreSQL via Python services.
4. **Which tools may run** — role checks again in the authorize node, then again inside each tool. Arguments are validated in Python.
5. **Whether a write happens** — a `PendingAction` row is prepared, LangGraph `interrupt`s, and `execute_pending` re-checks authorization, expiry, and idempotency before calling the same write service the REST API uses.

OpenRouter (`OPENROUTER_API_KEY`, default model `openai/gpt-4o-mini`) is optional. With the key empty, Helios still answers from the deterministic briefing and the deterministic tool planner.

---

## 2. Implementation

### High-level architecture

```text
Next.js UI
    → FastAPI (JWT)
        → conversation memory (follow-up rewrite)
        → LangGraph
              ├─ classify_intent → authorize
              ├─ domain nodes (leave, eligibility, approvals, workforce, records, routing)
              ├─ tool_orchestrate (fixed tool catalog; planner or model)
              ├─ hybrid policy retrieval (BM25 + dense + RRF)
              ├─ compose_response (deterministic narration; optional grounded explanation)
              └─ human_gate interrupt → execute_pending / cancel_pending
        → PostgreSQL (+ pgvector)  and  SQLite LangGraph checkpoints
```

| Layer | What is actually used |
| --- | --- |
| Frontend | Next.js, React, TypeScript, Tailwind. Token stored in `localStorage`. |
| Backend | FastAPI, Pydantic settings, SQLAlchemy |
| Orchestration | LangGraph `StateGraph` with conditional edges, `RetryPolicy` on classify / tools / compose, `interrupt` on the human gate |
| Checkpoints | `SqliteSaver` at `CHECKPOINT_PATH`. Compose sets `CHECKPOINT_STORE=sqlite` on the `helios-data` volume. `CHECKPOINT_STORE=postgres` switches to `PostgresSaver`. |
| Database | PostgreSQL 16 + pgvector (Compose). SQLite is an optional laptop `DATABASE_URL`. |
| Embeddings | Local `minishlab/potion-retrieval-32M` (512-d) via model2vec. Chat completion is never used to embed. An OpenAI-compatible embeddings endpoint is a separate setting. |
| Chat LLM | Optional OpenRouter. Transient failures retry; a bad or ungrounded answer is dropped. |
| Auth | PBKDF2-HMAC-SHA256 (60,000 iterations), JWT HS256, 12-hour TTL. Roles: `employee`, `manager`, `hr_admin`. |
| Clock | `ORG_CLOCK=2026-11-07` in Compose. Leave, holidays, and absence windows use this date. |

### Request path

1. **Memory** (`app/ai/memory.py`) loads the user’s conversation. If the turn looks like a follow-up, OpenRouter rewrites it into a standalone question when a key is set. Otherwise cue matching (`it`, `that month`, `the first one`, duration changes, and similar) is used. Workflow context (`record_context`) is inherited only for the same user and employee. Security still sees the raw text.
2. **`classify_intent`** runs `assess_security` first. Denials, organization-wide asks from a manager or employee, destructive requests, out-of-hierarchy asks, other-employee private data, and bulk private exports become `unauthorized_access` or `unsupported_mutation` before retrieval. Otherwise `rule_intent` runs, then specialized follow-up rules (approvals, eligibility, date recommendations, delayed approval, routing, workforce corrective). `QuerySpec` (`app/ai/query_spec.py`) interprets domain, scope, metric, and dates with no model and no authorization decision; it fills a missing intent or replaces a weak `general_hr` / `out_of_scope` label. `classify_with_model` runs only when the intent is still empty or still a weak catch-all, and the label must be in `ALLOWED_INTENTS`.
3. **`authorize`** repeats the security assessment and applies role gates: team and approval scope, self-only leave planning, no employee self-approval, named people must be direct reports for a manager, compensation lookup requires a named employee and blocks bulk export. Denial stops at `compose_response`.
4. **Work** is either a dedicated node or `tool_orchestrate`:
   - `employee_context` → `leave_analysis` → `retrieve_policy` for leave planning and requests.
   - `assess_eligibility` for advance leave-eligibility questions.
   - `approval_decision` for approve/reject (same resolution and `decide_approval` service as `POST /api/approvals/{id}/decide`).
   - `workforce_analytics` for HR absence analytics (department, month, office, z-score anomalies). Month-contribution follow-ups stay on the org-attendance tool path.
   - `lookup_records` for profile, directory, roster, team analytics, org leave, compensation, HR metrics, leave status, cancellation.
   - `investigate_then_retrieve` for an employee’s own routing (line manager vs approval manager). HR org-wide mismatch audits use `investigate_routing_discrepancies`.
   - `tool_orchestrate` for balances, attendance, holidays, team conflict, org attendance, transfers, and policy questions. Up to three tool rounds. The model may select tools only when `prefer_llm` is on. Org attendance, approvals, transfers, leave balance, personal attendance, team conflict, and the approval inbox use the deterministic planner so the model cannot change the filters behind a count.
5. **`prepare_workflow`** creates a pending action only when the turn is actually a write: leave submit / draft confirmation, an explicit routing-correction request, or an explicit HR corrective-recommendation request. Investigating a mismatch does not open the gate. Date recommendations are guidance and skip the gate.
6. **`compose_response`** always builds the briefing with `narrate`. For knowledge and tool-explanation intents it may call `generate_grounded_answer`. High-stakes payloads stay on the briefing: team overlap, approval inbox and decision text, routing audits, org absence counts, date recommendations, delayed-approval advice, and any turn that already has a pending action.
7. **`human_gate`** calls LangGraph `interrupt` when `pending_action` is set. Resume with confirm routes to `execute_workflow`; decline routes to `cancel_workflow`. Without a pending action the gate is skipped. The UI can also confirm through `POST /api/assistant/confirm`. Short replies such as “yes” / “confirm” and “cancel” are recognized in `app/ai/runtime.py`.

### Tools

The catalog in `app/ai/tools.py` is fixed. Each spec lists roles and whether it mutates. Handlers call existing services after `sanitize_tool_arguments`. The model cannot add a tool or emit SQL.

| Tool | Role |
| --- | --- |
| `get_leave_balance` | Signed-in employee’s annual/sick balance |
| `evaluate_leave_feasibility` | Balance, notice, conflicts for a window |
| `create_leave_draft` | Draft with holidays, projects, approvals; no submit |
| `submit_leave_request` | Pending submit until confirmation |
| `get_attendance` | Signed-in attendance window |
| `analyze_team_coverage` | Overlapping leave and coverage for authorized reports |
| `get_approval_inbox` | Pending items in the actor’s scope |
| `execute_approval` / `execute_rejection` | Prepare a decision; write only after confirmation |
| `department_attendance_analytics` | Scoped absence analytics |
| `compare_date_ranges` | Window comparison from stored rows |
| `investigate_employee_discrepancy` | Routing mismatch (org-wide tool is HR-scoped) |
| `search_hr_policies` | Hybrid retrieval |

### Human confirmation

Pending rows live in PostgreSQL (`pending_actions`), separate from the LangGraph thread. `create_or_reuse_pending` is idempotent. `execute_pending` locks the row, expires stale actions, refuses a second execution, and calls `revalidate_pending` (owner, role, live balance, live approval assignment, live routing target) before the write.

| Action type | What confirmation does |
| --- | --- |
| `submit_leave` | `submit_leave`. Insufficient balance is blocked and the balance is not reduced. Status becomes pending manager or pending HR. |
| `decide_approval` | `decide_approval` for leave or for a routing data-correction. Rejection reasons are stored. Audit is written. |
| `file_routing_correction` | Files a People Operations `data_correction` approval when line manager and approval manager still differ. A stale target is not applied. HR may file for the subject employee; other roles only for themselves. |
| `record_hr_corrective_recommendation` | HR-only audit entry. Advisory. Employee and attendance rows are not changed. |

Chat and REST share `decide_approval`. A missing durable pending id does not fall back to an in-memory write.

### Policy retrieval

Documents are seeded from `backend/app/knowledge/policies/` with version, effective and expiry dates, department, location, access (`public`, `internal`, `hr_confidential`), supersedes, and optional `conflict_group`. Expired, superseded, draft, and not-yet-effective documents stay in the database for audit and are excluded from retrieval. Department and location filters apply to non-HR roles.

For a query:

1. BM25 over visible chunks (in process).
2. Dense search: pgvector cosine (`<=>`) on PostgreSQL; in-Python cosine otherwise. Chunks from a different embedding model are skipped.
3. Reciprocal rank fusion (`k = 60`) of the lexical and dense lists.
4. Optional cross-encoder rerank (`RERANK_ENABLED`, default off). A rerank failure is reported and does not pretend fusion did not run.
5. If one side of a current conflict group is actually on topic, the other current side is attached. The answer discloses the conflict and does not pick a winner.
6. If the embedder fails or no chunk matches the active model, mode is `degraded_bm25` and the response says so.

A short synonym expansion (PTO, WFH, doctor’s note, overtime premium, transfer) is applied to the search string only. It does not generate policy text.

`POST /api/retrieval/evaluate` runs the retrieval checks in `app/ai/eval.py`.

### Grounded answers

`generate_grounded_answer` sends a JSON evidence pack: safe employee fields (no salary or performance unless that path already loaded them), tool payloads, citations, conflicts, retrieval mode, and a short memory summary. The model must reply with JSON (`answer`, `summary`, recommendations, cited policy ids). `grounding_errors` rejects numbers that are not in the evidence and policy slugs that were not retrieved. On rejection, timeout, or a missing key, the user receives the deterministic briefing (or an explicit unavailable message on hard LLM failure paths). `polish` / `is_grounded` remain in `app/ai/llm.py` as a stricter legacy rewrite and are not what `compose_response` calls.

### Workflows the product actually covers

- **Employee leave** — balance, eligibility, date recommendations (no submit), holidays in window, project conflicts, approval chain, duration changes, draft preview, confirmation. Notice and the 15-day consecutive cap come from policy plus the leave engine.
- **Leave status** — own pending requests and delayed-approval follow-ups. Employees cannot approve, including their own request.
- **Attendance and holidays** — the signed-in employee’s records and the seeded calendar for their office.
- **Manager team** — overlap and coverage risk, roster, approval inbox, urgency and approve/reject guidance, chat approve/reject with confirmation. Scope is direct reports. Organization-wide absence analytics are denied.
- **Routing** — seeded mismatches where `manager_id` and `approval_manager_id` differ. Employees can inspect their own record. HR can audit org-wide and file a correction only when they explicitly ask to file, apply, or correct it.
- **HR analytics** — unplanned absence (`absent` + `sick` over stored weekday rows), department and office comparisons, month-over-month change, anomalies at z-score ≥ 1.5 on department-months with at least 20 scheduled days. Recommendations are computed from those rows. A corrective “action” records an advisory audit entry.
- **Compensation** — HR may look up a named employee. Bulk compensation or performance export is blocked, including for HR. The assistant does not accept an employee id typed into the message for self-service lookups.
- **Security probes** — privilege escalation, control bypass, secrets and system prompts, destructive SQL or record deletion, cross-user memory, scope widening, social engineering, and other-employee medical or salary asks are denied before retrieval. REST profile reads return 403 outside `app/access.py` rules.

### Workspace UI

`/` login, then dashboard, assistant (streaming chat at `POST /api/assistant/chat/stream`), leave, team, analytics, policies, holidays, approvals, audit log, and settings. Navigation is role-aware in the API even where the shell lists the same links. In-app notifications are written on leave submit and leave decision.

### Repository layout

| Path | Role |
| --- | --- |
| `backend/app/ai/graph.py` | LangGraph nodes and routing |
| `backend/app/ai/runtime.py` | Checkpoint compile, chat/resume, confirm/cancel phrases |
| `backend/app/ai/context.py` | Rules, follow-ups, approval parsing |
| `backend/app/ai/query_spec.py` | Structured interpretation, no model |
| `backend/app/ai/security.py`, `guard.py`, `access.py` | Deny layer and RBAC |
| `backend/app/ai/memory.py` | Per-user history and follow-up rewrite |
| `backend/app/ai/orchestrate.py`, `tools.py` | Planner, model tool choice, execution |
| `backend/app/ai/retrieval.py`, `fusion.py`, `embeddings.py`, `ingest.py` | Hybrid RAG and indexing |
| `backend/app/ai/narrate.py`, `llm.py` | Briefing and grounded explanation |
| `backend/app/services/leave_engine.py`, `records.py`, `team_ops.py`, `hr_ops.py`, `analytics_engine.py` | Deterministic HR math |
| `backend/app/services/actions.py`, `confirmation.py`, `approval_resolution.py` | Writes, pending actions, approval targeting |
| `backend/app/knowledge/policies/` | Seeded policy corpus |
| `frontend/app/*` | Workspace pages |
| `docker-compose.yml` | Postgres, backend, frontend |

Startup is in `README.md`. Environment template: `.env.example`.

---

## 3. Technical decisions

| Decision | Rationale |
| --- | --- |
| Security assessment before intent | A benign clause must not route a message that also asks to bypass controls. |
| Rules, then `QuerySpec`, then the model | Demo and security routing stay stable. The model only labels gaps, and only with an allow-listed intent. |
| Deterministic services for HR math | Balances, headcount, absence rates, and z-scores are computed in Python. |
| Deterministic tool plans for counts and approvals | Stops the model from swapping a date filter or an approval target under a numeric answer. |
| Fixed tool catalog with per-call RBAC | The model selects; Python validates arguments and roles. |
| LangGraph `interrupt` plus a `PendingAction` row | The execute node is unreachable until resume, and the write is revalidated in Postgres (owner, expiry, idempotency, live record). |
| One `decide_approval` for chat and REST | The two channels cannot disagree on who may approve. |
| Hybrid BM25 + dense + RRF | Keyword policy titles and paraphrases both retrieve. Embeddings are a separate provider from chat. |
| Conflict groups disclosed, not resolved | Two current documents in force are both shown. |
| Grounding on numbers and cited slugs | Explanations can be prose; new figures and unseen policy ids are rejected. |
| Forced deterministic narration for overlap, approvals, and org counts | Those answers keep ids, pairs, and month figures intact. |
| Fixed `ORG_CLOCK` | Leave windows, attendance windows, and policy effective dates stay reproducible. |
| Local embeddings by default | The stack runs without a paid embedding API. |
| SQLite checkpoints in Compose | Graph resume does not share the Postgres pool with request traffic. `PostgresSaver` is implemented behind `CHECKPOINT_STORE=postgres`. |
| Advisory HR corrective record | The assistant can prepare a recommendation; it does not mutate attendance or employee rows to “fix” absence. |

---

## 4. Assumptions

1. Reviewers are evaluating an enterprise AI proof of concept, not a multi-tenant production HRIS.
2. Demo users are the seeded accounts. The shared password is documented in `README.md`.
3. Docker Desktop is the supported full-stack path. First start downloads the embedding model and seeds the database.
4. Attendance, leave, projects, policies, and holidays in Postgres are the system of record for the demo.
5. OpenRouter may be unset or out of credit. Classification, tools, and answers still run on the deterministic path.
6. “Today” for business logic is **2026-11-07** unless `ORG_CLOCK` changes.
7. Managers see direct reports. HR Admin has organization scope for analytics and confidential policy access. Employees see their own operational record.
8. Leave approval uses `approval_manager_id`, which can differ from `manager_id` after a transfer. That mismatch is a seeded scenario, not a live HRIS feed.
9. A routing correction files an approval request. It does not silently overwrite the reporting line.
10. Workforce corrective confirmation writes an audit recommendation only.

---

## 5. Limitations

1. **Synthetic organization.** There is no live HRIS, payroll, or identity provider. Writes land in this database (or, for corrective recommendations, in the audit log only).
2. **Routing is a large rule surface.** `context.py` and `query_spec.py` special-case phrasings (plan leave, overlaps, “contributed most”, pronoun follow-ups, delayed approval). Unusual wording can miss the intended node or hit a neighbouring rule.
3. **Follow-up memory is best-effort.** Without OpenRouter, only the cue list rewrites the question. A wrong rewrite still goes through security and RBAC, but it can attach the previous subject.
4. **Many answers stay templated on purpose.** Overlaps, approvals, routing audits, date recommendations, and org absence counts skip grounded generation. When generation does run, a failed number check returns the briefing, which reads like a template.
5. **Checkpoints are local to one backend process.** Compose uses SQLite on a volume. A second replica would not see the same LangGraph thread. Pending actions themselves are in Postgres, so a confirm can still execute if the pending id is known, but mid-graph resume is not shared. `CHECKPOINT_STORE=auto` also stays on SQLite.
6. **Scale is demo-sized.** Absence analytics load attendance rows for the scoped employees and aggregate in process. BM25 scores every visible policy chunk per query. Neither path is built for thousands of employees or a large corpus.
7. **Retrieval quality is bounded by the local model.** `potion-retrieval-32M` is a small static embedding model. Synonym expansion is a hand-written list. The cross-encoder is off by default, so near-miss paraphrases can stay on BM25.
8. **Calendars and policies are seeded.** Holiday applicability is by country, location, and department in the seed, not a government feed. Policy lifecycle (effective, expiry, supersedes) is only as current as those files.
9. **No automated test suite ships in this tree.** `backend/app/testing/` holds acceptance-isolation helpers, and retrieval evaluation is an API call. Regression coverage is the running Compose stack, not pytest.
10. **Session handling is POC-grade.** The JWT lasts 12 hours and the browser keeps it in `localStorage`. There is no SSO, refresh rotation, or server-side session revoke.
11. **Demo data drifts.** Approve, reject, submit, and routing-correction exercises change the seed. `docker compose down -v` is the clean reset.
12. **Manager analytics are direct-report only**, by design. Questions about a peer team or the whole company are denied rather than approximated.
13. **The tool list cannot grow at runtime.** A question outside the catalog and the rule set becomes a clarification, a policy retrieval, or `out_of_scope`.

---

## 6. Potential future improvements

1. Replace the regex cascade with a constrained classifier and entity extractor, and keep `assess_security` and `authorize` in front of it.
2. Turn on `CHECKPOINT_STORE=postgres` (already implemented) when more than one API process must resume the same thread, and add a lock around checkpoint setup.
3. Materialize monthly attendance aggregates so HR analytics do not scan raw rows on each question.
4. Put a real HRIS adapter behind `submit_leave`, `decide_approval`, and `file_routing_correction` without moving authorization or the confirmation gate into the model.
5. Add SSO, shorter-lived access tokens, and an httpOnly session cookie in place of `localStorage`.
6. Run `evaluate_retrieval` in CI on every policy-corpus change, and enable or replace the reranker once the eval set shows a gain.
7. Move reindex, bulk exports that HR is allowed to run, and audit retention onto background workers.
8. Ingest an official holiday feed and govern multi-country calendars the same way policy effective dates are governed.
9. Add a pytest suite around security denials, pending-action revalidation, and the deterministic tool plans that protect numeric answers.
10. Let grounded generation explain low-risk narratives while keeping a schema check on ids and rates for approvals and absence counts.
11. Expand the transfer-correction screen so HR can see the mismatch, the pending `data_correction`, and the later decision in one place.

---

## 7. Stakeholder trade-offs (as implemented)

| Trade-off | Helios stance |
| --- | --- |
| Automation vs oversight | Analysis, drafts, and recommendations are automatic. Submit, approve, reject, routing correction, and HR corrective notes wait for confirmation and are rechecked at execution. |
| Experience vs security | Self-service answers are immediate. Cross-employee private data, org-wide manager asks, and mixed jailbreak prompts are denied before retrieval. |
| Cost vs capability | The product runs with no model key. OpenRouter is limited to classify fallback, follow-up rewrite, tool choice on selected intents, and grounded explanation. |
| Faithful figures vs fluent prose | Counts, overlaps, and approvals stay on the deterministic briefing. Fluency is allowed only where grounding can check the numbers. |
| Maintainability vs integration depth | Services and the tool catalog are the boundary. The simulated writes are ordinary functions a later HRIS adapter can replace. |

---

## 8. How to evaluate quickly

1. Follow `README.md` (`docker compose up --build`). UI: `http://localhost:3000`. API: `http://localhost:8000/docs`.
2. Employee (`aisha.rahman@northstar.example`): leave balance, a December plan, holidays and projects, a one-day draft, then confirm. Ask for date recommendations and confirm no request is filed. Ask about a delayed approval on your own request.
3. Manager (`james.okonkwo@northstar.example`): overlapping leave for direct reports, approval inbox, then chat approve or reject and confirm. An organization-wide absence question should be denied.
4. HR (`helena.vogel@northstar.example`): department, month, and office absence analytics, including an anomaly if the window supports one. Ask to record a corrective recommendation and confirm; attendance rows should be unchanged. Ask about a transferred employee’s routing and file a correction only when you explicitly request it.
5. Security: ask any role for another employee’s salary or medical leave, and ask a manager for company-wide absenteeism. The assistant should deny before substituting someone else’s record. A direct profile API call outside scope should return 403.

Password for every demo account is in `README.md`. Reset the world with `docker compose down -v`.
