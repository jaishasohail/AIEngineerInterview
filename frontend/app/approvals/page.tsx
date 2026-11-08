"use client";

import { useEffect, useState } from "react";
import { api, pct } from "@/lib/api";
import { Badge, Button, Card, Empty, ErrorNote, PageHeader, statusTone, TextArea } from "@/components/ui";

type Approval = {
  id: number;
  action_type: string;
  stage: number;
  status: string;
  subject: string | null;
  can_decide: boolean;
  decision_note: string | null;
  payload: Record<string, unknown> | null;
  impact: Record<string, unknown> | null;
  leave: { start_date: string; end_date: string; business_days: number; status: string; reason: string; hr_required: boolean } | null;
};

export default function ApprovalsPage() {
  const [rows, setRows] = useState<Approval[]>([]);
  const [notes, setNotes] = useState<Record<number, string>>({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<number | null>(null);
  const [ready, setReady] = useState(false);

  async function load() {
    const body = await api<{ approvals: Approval[] }>("/api/approvals?status=all");
    setRows(body.approvals);
  }

  useEffect(() => {
    load().catch((err: Error) => setError(err.message)).finally(() => setReady(true));
  }, []);

  async function decide(id: number, decision: "approve" | "reject") {
    setBusy(id);
    setError("");
    try {
      await api(`/api/approvals/${id}/decide`, {
        method: "POST",
        body: JSON.stringify({ decision, note: notes[id] || "" }),
      });
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Decision failed");
    } finally {
      setBusy(null);
    }
  }

  return (
    <div>
      <PageHeader
        kicker="Approvals"
        title="Human decisions"
        text="Helios can prepare a request and describe the impact. Only the assigned manager or an HR administrator can change the simulated HRIS."
      />
      {error ? <div className="mb-4"><ErrorNote message={error} /></div> : null}
      {!ready && !error ? <p className="mb-4 text-sm text-ink/60">Loading approvals…</p> : null}
      {ready && rows.length === 0 ? <Empty>No approvals are visible to this account.</Empty> : null}
      <div className="space-y-4">
        {rows.map((row) => {
          const impact = (row.impact || {}) as { available_pct?: number; risk_level?: string; risks?: string[]; label?: string };
          return (
            <Card key={row.id} className="p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="font-serif text-2xl">{row.subject || "Request"}</h2>
                  <p className="text-sm text-ink/60">{row.action_type.replaceAll("_", " ")} · stage {row.stage}</p>
                </div>
                <Badge tone={statusTone(row.status)}>{row.status.replaceAll("_", " ")}</Badge>
              </div>
              {row.leave ? (
                <p className="mt-3 text-sm">{row.leave.start_date} → {row.leave.end_date} · {row.leave.business_days} business days · {row.leave.reason}</p>
              ) : null}
              {row.payload && row.action_type === "data_correction" ? (
                <p className="mt-3 text-sm">Move approval routing from {String(row.payload.from_name)} to {String(row.payload.to_name)}.</p>
              ) : null}
              {impact.risk_level ? (
                <div className="mt-3 rounded-xl bg-sand p-3 text-sm">
                  <p className="font-semibold">Deterministic impact · {impact.risk_level}{impact.available_pct !== undefined ? ` · availability ${pct(impact.available_pct)}` : ""}</p>
                  <ul className="mt-2 list-disc pl-5">
                    {(impact.risks || []).map((risk) => <li key={risk}>{risk}</li>)}
                  </ul>
                </div>
              ) : null}
              {row.can_decide ? (
                <div className="mt-4 space-y-3">
                  <TextArea rows={2} placeholder="Decision note" value={notes[row.id] || ""} onChange={(event) => setNotes({ ...notes, [row.id]: event.target.value })} />
                  <div className="flex gap-2">
                    <Button disabled={busy === row.id} onClick={() => void decide(row.id, "approve")}>Approve</Button>
                    <Button variant="danger" disabled={busy === row.id} onClick={() => void decide(row.id, "reject")}>Reject</Button>
                  </div>
                </div>
              ) : row.decision_note ? <p className="mt-3 text-sm text-ink/70">Note: {row.decision_note}</p> : null}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
