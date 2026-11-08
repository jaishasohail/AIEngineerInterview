"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, readUser } from "@/lib/api";
import { Badge, Button, Card, Empty, ErrorNote, statusTone } from "@/components/ui";

type Citation = { policy_id: string; title: string; version: string; effective_date: string; excerpt: string };
type PendingOption = { id: string; label: string; risk_level?: string; hr_required?: boolean };
type Turn = {
  conversation_id: number;
  message: string;
  intent?: string;
  decision_summary?: string;
  confidence?: string;
  citations?: Citation[];
  tool_trace?: { tool: string; status: string; summary: string }[];
  workflow?: { node: string; label: string; status: string; detail?: string }[];
  pending_action?: { type: string; confirmation_prompt: string; options: Record<string, PendingOption> } | null;
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
    "What is Noah Patel's salary and last performance rating?",
  ],
  manager: [
    "Three employees have requested leave during the same week. We have a critical project delivery coming up. Analyze the impact and suggest an approach.",
    "What notice period applies before taking annual leave?",
  ],
  hr_admin: [
    "Analyze employee absence trends across departments over the last three months.",
    "What is the overtime premium and after how many hours is it paid?",
    "When is People Operations co-approval required for leave?",
  ],
};

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
  }

  async function send(text: string, conversationId = activeId) {
    const message = text.trim();
    if (!message) return;
    followLatest.current = true;
    setBusy(true);
    setError("");
    setDraft("");
    setMessages((current) => [...current, { id: Date.now(), role: "user", content: message }]);
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
      setError(err instanceof Error ? err.message : "The assistant could not answer.");
    } finally {
      setBusy(false);
    }
  }

  async function confirm(optionId: string, confirmed: boolean) {
    if (!activeId) return;
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

  const last = [...messages].reverse().find((item) => item.role === "assistant")?.payload;

  return (
    <div className="grid min-h-0 w-full flex-1 grid-rows-[minmax(0,9.5rem)_minmax(0,1fr)_minmax(0,11rem)] gap-4 overflow-hidden xl:grid-cols-[230px_minmax(0,1fr)_320px] xl:grid-rows-1">
      <Card className="flex h-full min-h-0 flex-col overflow-hidden p-3">
        <Button className="mb-3 w-full shrink-0" variant="ghost" onClick={() => { setActiveId(null); setMessages([]); }}>
          New conversation
        </Button>
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto">
          {conversations.map((item) => (
            <button
              key={item.id}
              onClick={() => openConversation(item.id).catch((err: Error) => setError(err.message))}
              className={`w-full rounded-xl px-3 py-2 text-left text-sm ${activeId === item.id ? "bg-sand" : "hover:bg-sand/70"}`}
            >
              <span className="line-clamp-2 font-semibold">{item.title}</span>
              <span className="text-xs text-ink/50">{item.status.replaceAll("_", " ")}</span>
            </button>
          ))}
        </div>
      </Card>
      <Card className="flex h-full min-h-0 flex-col overflow-hidden">
        <div ref={scroller} onScroll={onConversationScroll} className="min-h-0 flex-1 space-y-4 overflow-y-auto overscroll-y-contain p-5">
          {messages.length === 0 ? (
            <div>
              <p className="font-serif text-3xl">What should Helios work through?</p>
              <div className="mt-4 grid gap-2">
                {(SUGGESTIONS[role] || SUGGESTIONS.employee).map((prompt) => (
                  <button key={prompt} className="rounded-2xl border border-line px-4 py-3 text-left text-sm hover:border-copper" onClick={() => void send(prompt, null)}>
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
            </article>
          ))}
          {busy ? <p className="text-sm text-ink/50">Running the workflow…</p> : null}
        </div>
        <form
          className="shrink-0 border-t border-line p-4"
          onSubmit={(event) => {
            event.preventDefault();
            void send(draft);
          }}
        >
          {error ? <div className="mb-3"><ErrorNote message={error} /></div> : null}
          <div className="flex gap-2">
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask about leave, policy, your team, or a reporting change"
              className="min-w-0 flex-1 rounded-full border border-line px-4 py-3 text-sm outline-none ring-copper/30 focus:ring-2"
            />
            <Button className="shrink-0" disabled={busy} type="submit">Send</Button>
          </div>
        </form>
      </Card>
      <div className="h-full min-h-0 space-y-4 overflow-y-auto overscroll-y-contain">
        <Card className="p-4">
          <h2 className="font-serif text-2xl">Decision</h2>
          {last?.decision_summary ? <p className="mt-2 text-sm leading-6">{last.decision_summary}</p> : <Empty>The latest answer will summarize the decision here.</Empty>}
          <div className="mt-3 flex flex-wrap gap-2">
            {last?.intent ? <Badge>{last.intent.replaceAll("_", " ")}</Badge> : null}
            {last?.confidence ? <Badge tone={statusTone(last.confidence)}>{last.confidence}</Badge> : null}
            {last?.orchestration_mode ? <Badge tone="ink">{last.orchestration_mode}</Badge> : null}
          </div>
        </Card>
        {last?.pending_action ? (
          <Card className="p-4">
            <h2 className="font-serif text-2xl">Confirmation</h2>
            <p className="mt-2 text-sm leading-6">{last.pending_action.confirmation_prompt}</p>
            <div className="mt-3 space-y-2">
              {Object.values(last.pending_action.options).map((option) => (
                <Button key={option.id} className="w-full" variant={option.id === "primary" ? "copper" : "ghost"} disabled={busy} onClick={() => void confirm(option.id, true)}>
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
            {(last?.citations || []).map((citation) => (
              <div key={citation.policy_id} className="rounded-xl bg-sand p-3 text-xs leading-5">
                <p className="font-semibold">{citation.title} v{citation.version}</p>
                <p className="mt-1 text-ink/70">{citation.excerpt}</p>
              </div>
            ))}
            {!last?.citations?.length ? <Empty>Grounded answers cite policy excerpts.</Empty> : null}
          </div>
        </Card>
      </div>
    </div>
  );
}
