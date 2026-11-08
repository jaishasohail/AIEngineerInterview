"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, pct, readUser } from "@/lib/api";
import { Badge, Card, Empty, ErrorNote, PageHeader, statusTone } from "@/components/ui";

type Dashboard = {
  org_clock: string;
  scope: string;
  headcount: number;
  out_of_office_next_14_days: number;
  pending_approvals: number;
  attendance_rate: number;
  departments: { department: string; headcount: number; location: string }[];
  monthly_absence: { month: string; unplanned_absence_rate: number }[];
  pending_preview: { id: number; action_type: string; status: string; subject: string | null; created_at: string }[];
};

const PROMPTS = [
  { role: "employee", label: "Plan two weeks off", q: "I'm planning to take two weeks off next month. Can you check my remaining balance, identify project conflicts, and tell me what approvals I need?" },
  { role: "manager", label: "Review overlapping leave", q: "Three employees have requested leave during the same week. We have a critical project delivery coming up. Analyze the impact and suggest an approach." },
  { role: "hr_admin", label: "Absence trends", q: "Analyze employee absence trends across departments over the last three months." },
  { role: "employee", label: "Check a stale manager route", q: "I changed departments and my reporting manager changed, but my leave requests are still going to my previous manager." },
];

export default function DashboardPage() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [error, setError] = useState("");
  const role = readUser()?.role;

  useEffect(() => {
    api<Dashboard>("/api/dashboard").then(setData).catch((err: Error) => setError(err.message));
  }, []);

  const prompts = PROMPTS.filter((item) => !role || item.role === role || (role === "hr_admin" && item.role !== "manager"));

  return (
    <div>
      <PageHeader
        kicker="Northstar Global"
        title="Workforce at a glance"
        text="Figures come from the simulated HRIS. The organization clock stays on 7 November 2026 so the Atlas deadline and December leave cluster remain reproducible."
      />
      {error ? <ErrorNote message={error} /> : null}
      {!data && !error ? <p className="text-sm text-ink/60">Loading the HRIS extract…</p> : null}
      {data ? (
        <>
          <div className="grid gap-4 md:grid-cols-4">
            <Stat label={data.scope === "hr_admin" ? "Employees" : "People in scope"} value={String(data.headcount)} />
            <Stat label="Out in the next 14 days" value={String(data.out_of_office_next_14_days)} />
            <Stat label="Pending approvals you can see" value={String(data.pending_approvals)} />
            <Stat label="Attendance, last 30 days" value={pct(data.attendance_rate)} />
          </div>
          <div className="mt-5 grid gap-4 xl:grid-cols-[1.3fr_0.7fr]">
            <Card className="min-w-0 overflow-hidden p-5">
              <h2 className="font-serif text-2xl">Headcount</h2>
              <div className="mt-4 h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.departments}>
                    <CartesianGrid stroke="#E5DCD0" vertical={false} />
                    <XAxis dataKey="department" tick={{ fontSize: 11 }} interval={0} angle={-18} height={60} />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Bar dataKey="headcount" fill="#1F6B5A" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
            <Card className="p-5">
              <h2 className="font-serif text-2xl">Waiting on a person</h2>
              <div className="mt-4 space-y-3">
                {data.pending_preview.length === 0 ? <Empty>Nothing is waiting in your queue.</Empty> : null}
                {data.pending_preview.map((item) => (
                  <Link key={item.id} href="/approvals" className="block rounded-xl border border-line px-3 py-3 hover:border-copper">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-semibold">{item.subject || item.action_type}</p>
                      <Badge tone={statusTone(item.status)}>{item.status.replaceAll("_", " ")}</Badge>
                    </div>
                    <p className="mt-1 text-xs text-ink/60">{item.action_type.replaceAll("_", " ")}</p>
                  </Link>
                ))}
              </div>
            </Card>
          </div>
          <Card className="mt-4 p-5">
            <h2 className="font-serif text-2xl">Ask Helios</h2>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              {prompts.map((prompt) => (
                <Link key={prompt.label} href={`/assistant?q=${encodeURIComponent(prompt.q)}`} className="rounded-2xl border border-line px-4 py-4 hover:border-copper">
                  <p className="text-sm font-semibold">{prompt.label}</p>
                  <p className="mt-1 line-clamp-2 text-xs leading-5 text-ink/60">{prompt.q}</p>
                </Link>
              ))}
            </div>
          </Card>
        </>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Card className="p-4">
      <p className="text-xs uppercase tracking-[0.14em] text-ink/50">{label}</p>
      <p className="mt-2 font-serif text-4xl">{value}</p>
    </Card>
  );
}
