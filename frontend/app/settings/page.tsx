"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Badge, Button, Card, ErrorNote, PageHeader } from "@/components/ui";

type Settings = {
  profile: { full_name: string; email: string; role: string; job_title: string; department: string | null; location: string };
  org_clock: string;
  llm_enabled: boolean;
  model: string | null;
  embedding_mode: string;
  database: string;
  annual_remaining: number | null;
};

type EvalResult = {
  precision_at_k: number;
  recall_at_k: number;
  mrr: number;
  results: { question: string; retrieved: string[]; recall: number; confidence: string; forbidden_hits: string[] }[];
};

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [evaluation, setEvaluation] = useState<EvalResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api<Settings>("/api/settings").then(setSettings).catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div>
      <PageHeader
        kicker="Settings"
        title="How this workspace is running"
        text="The model key stays on the server. This screen only reports whether OpenRouter is configured."
      />
      {error ? <div className="mb-4"><ErrorNote message={error} /></div> : null}
      {settings ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="p-5">
            <h2 className="font-serif text-2xl">{settings.profile.full_name}</h2>
            <p className="mt-1 text-sm text-ink/70">{settings.profile.job_title} · {settings.profile.department} · {settings.profile.location}</p>
            <dl className="mt-4 space-y-2 text-sm">
              <Row label="Email" value={settings.profile.email} />
              <Row label="Role" value={settings.profile.role} />
              <Row label="Organization clock" value={settings.org_clock} />
              <Row label="Annual days remaining" value={settings.annual_remaining === null ? "—" : String(settings.annual_remaining)} />
              <Row label="Database" value={settings.database} />
              <Row label="Retrieval" value={settings.embedding_mode} />
            </dl>
          </Card>
          <Card className="p-5">
            <h2 className="font-serif text-2xl">Model</h2>
            <div className="mt-3 flex items-center gap-2">
              <Badge tone={settings.llm_enabled ? "pine" : "copper"}>{settings.llm_enabled ? "OpenRouter connected" : "Deterministic mode"}</Badge>
            </div>
            <p className="mt-3 text-sm leading-6 text-ink/70">
              {settings.llm_enabled
                ? `Responses may be rewritten by ${settings.model}. Any number that is not in the computed briefing is discarded.`
                : "No API key is set. Intent routing, retrieval, analytics, and workflows still run. Narrative text is assembled from the tool results."}
            </p>
            {settings.profile.role === "hr_admin" ? (
              <Button
                className="mt-4"
                disabled={busy}
                onClick={() => {
                  setBusy(true);
                  api<EvalResult>("/api/retrieval/evaluate", { method: "POST" })
                    .then(setEvaluation)
                    .catch((err: Error) => setError(err.message))
                    .finally(() => setBusy(false));
                }}
              >
                {busy ? "Scoring retrieval…" : "Run retrieval evaluation"}
              </Button>
            ) : null}
          </Card>
        </div>
      ) : null}
      {evaluation ? (
        <Card className="mt-4 p-5">
          <h2 className="font-serif text-2xl">Retrieval evaluation</h2>
          <p className="mt-2 text-sm">Precision@K {evaluation.precision_at_k} · Recall@K {evaluation.recall_at_k} · MRR {evaluation.mrr}</p>
          <div className="mt-4 space-y-3">
            {evaluation.results.map((row) => (
              <div key={row.question} className="rounded-xl bg-sand p-3 text-sm">
                <p className="font-semibold">{row.question}</p>
                <p className="mt-1 text-xs text-ink/70">Retrieved {row.retrieved.join(", ") || "nothing"} · recall {row.recall} · {row.confidence}</p>
              </div>
            ))}
          </div>
        </Card>
      ) : null}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 border-b border-line py-2">
      <dt className="text-ink/50">{label}</dt>
      <dd className="text-right font-semibold">{value}</dd>
    </div>
  );
}
