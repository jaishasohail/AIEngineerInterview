"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Badge, Button, Card, Empty, ErrorNote, Field, Input, PageHeader, statusTone, TextArea } from "@/components/ui";

type Balance = { leave_type: string; entitled: number; used: number; pending: number; remaining: number; year: number };
type LeaveRow = {
  id: number;
  employee_name: string;
  start_date: string;
  end_date: string;
  business_days: number;
  status: string;
  reason: string;
  routed_to: string | null;
  hr_required: boolean;
};

export default function LeavePage() {
  const [balances, setBalances] = useState<Balance[]>([]);
  const [requests, setRequests] = useState<LeaveRow[]>([]);
  const [start, setStart] = useState("2026-12-07");
  const [end, setEnd] = useState("2026-12-18");
  const [reason, setReason] = useState("Planned annual leave");
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);

  async function load() {
    const [balanceBody, requestBody] = await Promise.all([
      api<{ balances: Balance[] }>("/api/leave/balances"),
      api<{ requests: LeaveRow[] }>("/api/leave/requests"),
    ]);
    setBalances(balanceBody.balances);
    setRequests(requestBody.requests);
  }

  useEffect(() => {
    load().catch((err: Error) => setError(err.message)).finally(() => setReady(true));
  }, []);

  return (
    <div>
      <PageHeader
        kicker="Leave"
        title="Balances and requests"
        text="Submitting here is a human action. The service recomputes notice, balance, project conflicts, and the approval route before anything is reserved."
      />
      {error ? <ErrorNote message={error} /> : null}
      {note ? <p className="mb-4 rounded-xl bg-pine/10 px-3 py-2 text-sm text-pine">{note}</p> : null}
      <div className="grid gap-4 md:grid-cols-2">
        {balances.map((balance) => (
          <Card key={balance.leave_type} className="p-5">
            <p className="text-xs uppercase tracking-[0.14em] text-ink/50">{balance.leave_type} · {balance.year}</p>
            <p className="mt-2 font-serif text-4xl">{balance.remaining} days left</p>
            <p className="mt-2 text-sm text-ink/70">Entitled {balance.entitled} · used {balance.used} · pending {balance.pending}</p>
          </Card>
        ))}
      </div>
      <Card className="mt-4 p-5">
        <h2 className="font-serif text-2xl">File annual leave</h2>
        <form
          className="mt-4 grid gap-4 md:grid-cols-2"
          onSubmit={async (event) => {
            event.preventDefault();
            setBusy(true);
            setError("");
            setNote("");
            try {
              const result = await api<{ message: string }>("/api/leave/requests", {
                method: "POST",
                body: JSON.stringify({ start_date: start, end_date: end, leave_type: "annual", reason }),
              });
              setNote(result.message);
              await load();
            } catch (err) {
              setError(err instanceof Error ? err.message : "Could not file leave");
            } finally {
              setBusy(false);
            }
          }}
        >
          <Field label="Start"><Input type="date" value={start} onChange={(event) => setStart(event.target.value)} required /></Field>
          <Field label="End"><Input type="date" value={end} onChange={(event) => setEnd(event.target.value)} required /></Field>
          <div className="md:col-span-2">
            <Field label="Reason"><TextArea value={reason} onChange={(event) => setReason(event.target.value)} rows={3} /></Field>
          </div>
          <Button disabled={busy} type="submit">{busy ? "Checking rules…" : "Submit for approval"}</Button>
        </form>
      </Card>
      <Card className="mt-4 overflow-hidden">
        <div className="border-b border-line px-5 py-4">
          <h2 className="font-serif text-2xl">Requests in your scope</h2>
        </div>
        {!ready ? <div className="p-5 text-sm text-ink/60">Loading requests…</div> : requests.length === 0 ? <div className="p-5"><Empty>No leave requests yet.</Empty></div> : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-ink/50">
                <tr>
                  <th className="px-5 py-3">Person</th>
                  <th className="px-5 py-3">Dates</th>
                  <th className="px-5 py-3">Days</th>
                  <th className="px-5 py-3">Route</th>
                  <th className="px-5 py-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {requests.map((row) => (
                  <tr key={row.id} className="border-t border-line">
                    <td className="px-5 py-3 font-semibold">{row.employee_name}</td>
                    <td className="px-5 py-3">{row.start_date} → {row.end_date}</td>
                    <td className="px-5 py-3">{row.business_days}</td>
                    <td className="px-5 py-3">{row.routed_to || "—"}{row.hr_required ? " + HR" : ""}</td>
                    <td className="px-5 py-3"><Badge tone={statusTone(row.status)}>{row.status.replaceAll("_", " ")}</Badge></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
