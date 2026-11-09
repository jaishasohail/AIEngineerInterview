"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, readUser } from "@/lib/api";
import { Badge, Button, Card, Empty, ErrorNote, statusTone } from "@/components/ui";

type Citation = {
  policy_id: string;
  title: string;
  version: string;
  effective_date: string;
  excerpt: string;
  chunk_id?: number;
};
type PendingOption = { id: string; label: string; risk_level?: string; hr_required?: boolean };
type ToolStep = { tool: string; status: string; summary: string };
type WorkflowStep = { node: string; label: string; status: string; detail?: string };
type Turn = {
  conversation_id: number;
  message: string;
  intent?: string;
  decision_summary?: string;
  confidence?: string;
  citations?: Citation[];
  tool_trace?: ToolStep[];
  workflow?: WorkflowStep[];
  pending_action?: {
    type?: string;
    action_type?: string;
    confirmation_prompt?: string;
    options?: Record<string, PendingOption>;
    status?: string;
    pending_action_id?: number;
  } | null;
  analysis?: Record<string, unknown> | null;
  analytics?: Record<string, unknown> | null;
  orchestration_mode?: string;
  authorization?: { allowed: boolean; reason: string };
};

type StoredMessage = { id: number; role: string; content: string; payload?: Turn | null };
type Conversation = { id: number; title: string; status: string };

const SUGGESTIONS: Record<string, string[]> = {
  employee: [
    "I'm planning to take two weeks off next month. Can you check my remaining balance, identify project conflicts, and tell me what approvals I need?",
    "What is the overtime premium and after how many hours is it paid?",
    "I changed departments and my reporting manager changed, but my leave requests are still going to my previous manager.",
    "How many days was I present and absent in October?",
  ],
  manager: [
    "Show my pending approval inbox",
    "Three employees have requested leave during the same week. We have a critical project delivery coming up. Analyze the impact and suggest an approach.",
    "What notice period applies before taking annual leave?",
  ],
  hr_admin: [
    "Analyze employee absence trends across departments over the last three months.",
    "What is the overtime premium and after how many hours is it paid?",
    "When is People Operations co-approval required for leave?",
  ],
};

function outcomeFromTurn(turn: Turn | null | undefined): { label: string; tone: ReturnType<typeof statusTone> } | null {
  if (!turn) return null;
  if (turn.authorization && turn.authorization.allowed === false) {
    return { label: "not authorized", tone: "bad" };
  }
  const pending = turn.pending_action;
  if (pending && (pending.status === "awaiting_confirmation" || pending.confirmation_prompt)) {
    return { label: "awaiting confirmation", tone: "warn" };
  }
  const tools = turn.tool_trace || [];
  if (tools.some((item) => ["denied", "error", "blocked", "failed"].includes(String(item.status).toLowerCase()))) {
    return { label: "action failed", tone: "bad" };
  }
  const analysisStatus = String((turn.analysis as { status?: string } | null)?.status || "").toLowerCase();
  if (["blocked", "denied", "failed", "duplicate", "rejected"].includes(analysisStatus)) {
    return { label: analysisStatus, tone: "bad" };
  }
  if (["submitted", "approved", "executed"].includes(analysisStatus)) {
    return { label: analysisStatus, tone: "pine" };
  }
  const text = (turn.message || "").toLowerCase();
  if (text.includes("was rejected") || text.includes("not submitted") || text.includes("did not submit")) {
    return { label: text.includes("rejected") ? "rejected" : "cancelled", tone: text.includes("rejected") ? "bad" : "ink" };
  }
  if (text.includes("is submitted") || text.includes("is approved") || text.includes("were recorded")) {
    if (text.includes("approved")) return { label: "approved", tone: "pine" };
    if (text.includes("submitted")) return { label: "submitted", tone: "pine" };
  }
  if (turn.confidence === "denied") return { label: "not authorized", tone: "bad" };
  return null;
}

export default function AssistantPage() {
  return (
    <Suspense fallback={<p className="text-sm text-ink/60">Opening the assistant…</p>}>
      <Assistant />
    </Suspense>
  );
}

function Assistant() {
  const params = useSearchParams();
  const role = readUser()?.role || "employee";
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [messages, setMessages] = useState<StoredMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const scroller = useRef<HTMLDivElement>(null);
  const followLatest = useRef(true);
  const question = params.get("q");

  async function refreshList() {
    const body = await api<{ conversations: Conversation[] }>("/api/assistant/conversations");
    setConversations(body.conversations);
  }

  async function openConversation(id: number) {
    const body = await api<{ messages: StoredMessage[] }>(`/api/assistant/conversations/${id}`);
    followLatest.current = true;
    setActiveId(id);
    setMessages(body.messages);
    setError("");
  }

  async function send(text: string, conversationId: number | null = activeId) {
    const message = text.trim();
    if (!message || busy) return;
    followLatest.current = true;
    setBusy(true);
    setError("");
    setDraft("");
    const optimisticId = Date.now();
    setMessages((current) => [...current, { id: optimisticId, role: "user", content: message }]);
    try {
      const payload = await api<Turn>("/api/assistant/chat", {
        method: "POST",
        body: JSON.stringify({ message, conversation_id: conversationId }),
      });
      setActiveId(payload.conversation_id);
      setMessages((current) => [
        ...current,
        { id: Date.now() + 1, role: "assistant", content: payload.message, payload },
      ]);
      await refreshList();
    } catch (err) {
      setMessages((current) => current.filter((item) => item.id !== optimisticId));
      setError(err instanceof Error ? err.message : "The assistant could not answer.");
    } finally {
      setBusy(false);
    }
  }

  async function confirm(optionId: string, confirmed: boolean) {
    if (!activeId || busy) return;
    setBusy(true);
    setError("");
    try {
      const payload = await api<Turn>("/api/assistant/confirm", {
        method: "POST",
        body: JSON.stringify({ conversation_id: activeId, confirmed, option_id: optionId }),
      });
      setMessages((current) => [
        ...current,
        { id: Date.now(), role: "user", content: confirmed ? `Confirmed ${optionId}` : "Cancelled" },
        { id: Date.now() + 1, role: "assistant", content: payload.message, payload },
      ]);
      await refreshList();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Confirmation failed.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    refreshList().catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!question) return;
    let active = true;
    setBusy(true);
    setError("");
    setMessages([{ id: 1, role: "user", content: question }]);
    api<Turn>("/api/assistant/chat", {
      method: "POST",
      body: JSON.stringify({ message: question, conversation_id: null }),
    })
      .then((payload) => {
        if (!active) return;
        setActiveId(payload.conversation_id);
        setMessages([
          { id: 1, role: "user", content: question },
          { id: 2, role: "assistant", content: payload.message, payload },
        ]);
        setBusy(false);
        return refreshList();
      })
      .catch((err: Error) => {
        if (!active) return;
        setError(err.message);
        setBusy(false);
      });
    return () => {
      active = false;
    };
  }, [question]);

  function onConversationScroll() {
    const node = scroller.current;
    if (!node) return;
    followLatest.current = node.scrollHeight - node.scrollTop - node.clientHeight < 96;
  }

  useEffect(() => {
    const node = scroller.current;
    if (!node || !followLatest.current) return;
    node.scrollTo({ top: node.scrollHeight });
  }, [messages, busy]);

  const last = [...messages].reverse().find((item) => item.role === "assistant")?.payload || null;
  const outcome = outcomeFromTurn(last);
  const denied = last?.authorization?.allowed === false;
  const pending = last?.pending_action;
  const pendingOptions = pending?.options ? Object.values(pending.options) : [];
  const showConfirm = Boolean(pending?.confirmation_prompt && pendingOptions.length && !denied);

  const sidePanel = (
    <>
      <Card className="p-4">
        <h2 className="font-serif text-2xl">Decision</h2>
        {denied && last?.authorization?.reason ? (
          <p className="mt-2 rounded-xl bg-red-50 px-3 py-2 text-sm leading-6 text-red-800">{last.authorization.reason}</p>
        ) : last?.decision_summary ? (
          <p className="mt-2 text-sm leading-6">{last.decision_summary}</p>
        ) : (
          <Empty>The latest answer will summarize the decision here.</Empty>
        )}
        <div className="mt-3 flex flex-wrap gap-2">
          {outcome ? <Badge tone={outcome.tone}>{outcome.label}</Badge> : null}
          {last?.intent ? <Badge>{last.intent.replaceAll("_", " ")}</Badge> : null}
          {last?.confidence ? <Badge tone={statusTone(last.confidence)}>{last.confidence}</Badge> : null}
          {last?.orchestration_mode ? <Badge tone="ink">{last.orchestration_mode}</Badge> : null}
        </div>
      </Card>
      {showConfirm ? (
        <Card className="p-4">
          <h2 className="font-serif text-2xl">Confirmation</h2>
          <p className="mt-1 text-xs uppercase tracking-[0.14em] text-ink/50">
            {(pending?.action_type || pending?.type || "action").replaceAll("_", " ")} · nothing written yet
          </p>
          <p className="mt-2 text-sm leading-6">{pending?.confirmation_prompt}</p>
          <div className="mt-3 space-y-2">
            {pendingOptions.map((option) => (
              <Button
                key={option.id}
                className="w-full"
                variant={option.id === "primary" ? "copper" : "ghost"}
                disabled={busy}
                onClick={() => void confirm(option.id, true)}
              >
                {option.label}
              </Button>
            ))}
            <Button className="w-full" variant="ghost" disabled={busy} onClick={() => void confirm("primary", false)}>
              Do not submit
            </Button>
          </div>
        </Card>
      ) : null}
      <Card className="p-4">
        <h2 className="font-serif text-2xl">Tools</h2>
        <ol className="mt-3 space-y-2">
          {(last?.tool_trace || []).map((step, index) => (
            <li key={`${step.tool}-${index}`} className="text-sm">
              <span className="font-semibold">{step.tool.replaceAll("_", " ")}</span>
              <span className="text-ink/50"> · {step.status}</span>
              {step.summary ? <span className="block text-xs text-ink/60">{step.summary}</span> : null}
            </li>
          ))}
          {!last?.tool_trace?.length ? <Empty>Tool calls appear after a question.</Empty> : null}
        </ol>
      </Card>
      <Card className="p-4">
        <h2 className="font-serif text-2xl">Workflow</h2>
        <ol className="mt-3 space-y-2">
          {(last?.workflow || []).map((step, index) => (
            <li key={`${step.node}-${index}`} className="text-sm">
              <span className="font-semibold">{step.label}</span>
              <span className="text-ink/50"> · {step.status}</span>
              {step.detail ? <span className="block text-xs text-ink/60">{step.detail}</span> : null}
            </li>
          ))}
          {!last?.workflow?.length ? <Empty>Nodes appear after a question.</Empty> : null}
        </ol>
      </Card>
      <Card className="p-4">
        <h2 className="font-serif text-2xl">Sources</h2>
        <div className="mt-3 space-y-3">
          {(last?.citations || []).map((citation, index) => (
            <div key={`${citation.policy_id}-${citation.chunk_id ?? index}`} className="rounded-xl bg-sand p-3 text-xs leading-5">
              <p className="font-semibold">
                {citation.title} v{citation.version}
              </p>
              <p className="mt-1 text-ink/70">{citation.excerpt}</p>
            </div>
          ))}
          {!last?.citations?.length ? <Empty>Grounded answers cite policy excerpts.</Empty> : null}
        </div>
      </Card>
    </>
  );

  return (
    <div className="flex min-h-0 w-full flex-1 flex-col gap-4 overflow-hidden xl:grid xl:grid-cols-[230px_minmax(0,1fr)_320px]">
      <Card className="flex max-h-40 min-w-0 shrink-0 flex-col overflow-hidden p-3 xl:max-h-none xl:h-full">
        <Button
          className="mb-3 w-full shrink-0"
          variant="ghost"
          disabled={busy}
          onClick={() => {
            setActiveId(null);
            setMessages([]);
            setError("");
          }}
        >
          New conversation
        </Button>
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto">
          {conversations.map((item) => (
            <button
              key={item.id}
              disabled={busy}
              onClick={() => openConversation(item.id).catch((err: Error) => setError(err.message))}
              className={`w-full rounded-xl px-3 py-2 text-left text-sm ${activeId === item.id ? "bg-sand" : "hover:bg-sand/70"}`}
            >
              <span className="line-clamp-2 font-semibold">{item.title}</span>
              <span className="block text-xs text-ink/50">{item.status.replaceAll("_", " ")}</span>
            </button>
          ))}
        </div>
      </Card>
      <Card className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <div ref={scroller} onScroll={onConversationScroll} className="min-h-0 flex-1 space-y-4 overflow-y-auto overscroll-y-contain p-5">
          {messages.length === 0 ? (
            <div>
              <p className="font-serif text-3xl">What should Helios work through?</p>
              <div className="mt-4 grid gap-2">
                {(SUGGESTIONS[role] || SUGGESTIONS.employee).map((prompt) => (
                  <button
                    key={prompt}
                    disabled={busy}
                    className="rounded-2xl border border-line px-4 py-3 text-left text-sm hover:border-copper disabled:opacity-50"
                    onClick={() => void send(prompt, null)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : null}
          {messages.map((item) => (
            <article key={item.id} className={item.role === "user" ? "ml-8 min-w-0 rounded-2xl bg-ink px-4 py-3 text-sm text-sand" : "mr-4 min-w-0"}>
              {item.role === "assistant" ? <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-copper">Helios</p> : null}
              <p className="whitespace-pre-wrap break-words text-sm leading-6">{item.content}</p>
              {item.role === "assistant" && item.payload?.authorization?.allowed === false ? (
                <p className="mt-2 rounded-xl bg-red-50 px-3 py-2 text-xs text-red-800">
                  Authorization denied by the backend. No confidential records were returned.
                </p>
              ) : null}
            </article>
          ))}
          {busy ? (
            <p className="flex items-center gap-2 text-sm text-ink/50" aria-live="polite">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-copper" />
              Running the workflow…
            </p>
          ) : null}
        </div>
        {(outcome || showConfirm) && (
          <div className="shrink-0 border-t border-line bg-sand/40 px-4 py-3 xl:hidden">
            <div className="mb-2 flex flex-wrap gap-2">
              {outcome ? <Badge tone={outcome.tone}>{outcome.label}</Badge> : null}
              {last?.intent ? <Badge>{last.intent.replaceAll("_", " ")}</Badge> : null}
            </div>
            {denied && last?.authorization?.reason ? (
              <p className="mb-2 text-xs leading-5 text-red-800">{last.authorization.reason}</p>
            ) : null}
            {showConfirm ? (
              <div className="space-y-2">
                <p className="text-sm leading-5">{pending?.confirmation_prompt}</p>
                {pendingOptions.map((option) => (
                  <Button
                    key={option.id}
                    className="w-full"
                    variant={option.id === "primary" ? "copper" : "ghost"}
                    disabled={busy}
                    onClick={() => void confirm(option.id, true)}
                  >
                    {option.label}
                  </Button>
                ))}
                <Button className="w-full" variant="ghost" disabled={busy} onClick={() => void confirm("primary", false)}>
                  Do not submit
                </Button>
              </div>
            ) : null}
          </div>
        )}
        <form
          className="shrink-0 border-t border-line p-4"
          onSubmit={(event) => {
            event.preventDefault();
            void send(draft);
          }}
        >
          {error ? (
            <div className="mb-3">
              <ErrorNote message={error} />
            </div>
          ) : null}
          <div className="flex gap-2">
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              disabled={busy}
              placeholder="Ask about leave, policy, your team, or a reporting change"
              className="min-w-0 flex-1 rounded-full border border-line px-4 py-3 text-sm outline-none ring-copper/30 focus:ring-2 disabled:opacity-60"
            />
            <Button className="shrink-0" disabled={busy || !draft.trim()} type="submit">
              {busy ? "Working…" : "Send"}
            </Button>
          </div>
        </form>
      </Card>
      <div className="hidden min-h-0 min-w-0 space-y-4 overflow-y-auto overscroll-y-contain xl:block xl:h-full">{sidePanel}</div>
    </div>
  );
}
