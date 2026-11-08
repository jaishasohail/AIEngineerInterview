"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Badge, Card, Empty, ErrorNote, PageHeader, statusTone } from "@/components/ui";

type Log = {
  id: number;
  actor: string | null;
  action: string;
  resource_type: string;
  resource_id: string | null;
  outcome: string;
  details: Record<string, unknown> | null;
  created_at: string;
};

export default function AuditPage() {
  const [logs, setLogs] = useState<Log[]>([]);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    api<{ logs: Log[] }>("/api/audit?limit=150")
      .then((body) => setLogs(body.logs))
      .catch((err: Error) => setError(err.message))
      .finally(() => setReady(true));
  }, []);

  return (
    <div>
      <PageHeader
        kicker="Control"
        title="Audit log"
        text="Denied reads are recorded here as well as successful submissions and approvals. Salary values are scrubbed before they are stored."
      />
      {error ? <ErrorNote message={error} /> : null}
      <Card className="overflow-hidden">
        {!ready ? <div className="p-5 text-sm text-ink/60">Loading the audit log…</div> : logs.length === 0 ? <div className="p-5"><Empty>No audit rows for this account yet.</Empty></div> : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead className="text-xs uppercase tracking-wide text-ink/50">
                <tr>
                  <th className="px-4 py-3">When</th>
                  <th className="px-4 py-3">Actor</th>
                  <th className="px-4 py-3">Action</th>
                  <th className="px-4 py-3">Outcome</th>
                  <th className="px-4 py-3">Detail</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => (
                  <tr key={log.id} className="border-t border-line align-top">
                    <td className="px-4 py-3 whitespace-nowrap text-xs">{log.created_at.replace("T", " ").slice(0, 16)}</td>
                    <td className="px-4 py-3">{log.actor || "—"}</td>
                    <td className="px-4 py-3">{log.action}</td>
                    <td className="px-4 py-3"><Badge tone={statusTone(log.outcome)}>{log.outcome}</Badge></td>
                    <td className="px-4 py-3 text-xs text-ink/70">{log.details ? JSON.stringify(log.details) : log.resource_type}</td>
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
