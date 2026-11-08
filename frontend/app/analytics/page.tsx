"use client";

import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, pct } from "@/lib/api";
import { Badge, Card, ErrorNote, PageHeader } from "@/components/ui";

type Analytics = {
  period: { start: string; end: string };
  scope: string;
  method: string;
  org_unplanned_absence_rate: number;
  departments: { department: string; unplanned_absence_rate: number; leave_utilization: number; scheduled_days: number; unplanned_absent: number }[];
  monthly: { month: string; department: string; unplanned_absence_rate: number }[];
  anomalies: { department: string; month: string; unplanned_absence_rate: number; zscore: number; note: string }[];
  insights: string[];
  personal_monthly: { month: string; present: number; remote: number; leave: number; unplanned: number }[];
};

export default function AnalyticsPage() {
  const [data, setData] = useState<Analytics | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api<Analytics>("/api/analytics?months=3").then(setData).catch((err: Error) => setError(err.message));
  }, []);

  const monthly = data ? pivotMonthly(data.monthly) : [];
  const bars = (data?.departments || []).map((row) => ({
    department: row.department.replace(" Operations", "").replace(" & Compliance", ""),
    absence: Math.round(row.unplanned_absence_rate * 1000) / 10,
    utilization: Math.round((row.leave_utilization || 0) * 1000) / 10,
  }));

  return (
    <div>
      <PageHeader
        kicker="Analytics"
        title="Absence, calculated"
        text="Rates are computed in Python from attendance rows. Helios can explain this payload. It is not allowed to invent a second set of figures."
      />
      {error ? <ErrorNote message={error} /> : null}
      {!data && !error ? <p className="text-sm text-ink/60">Calculating…</p> : null}
      {data ? (
        <>
          <p className="mb-4 text-sm text-ink/70">
            {data.period.start} to {data.period.end} · scope {data.scope} · unplanned rate {pct(data.org_unplanned_absence_rate)}
          </p>
          <div className="grid gap-4 xl:grid-cols-2">
            <Card className="min-w-0 overflow-hidden p-5">
              <h2 className="font-serif text-2xl">Unplanned absence by department</h2>
              <div className="mt-4 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={bars}>
                    <CartesianGrid stroke="#E5DCD0" vertical={false} />
                    <XAxis dataKey="department" tick={{ fontSize: 11 }} interval={0} angle={-18} height={70} />
                    <YAxis unit="%" />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="absence" name="Unplanned %" fill="#9C5B3C" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="utilization" name="Leave used %" fill="#1F6B5A" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
            <Card className="min-w-0 overflow-hidden p-5">
              <h2 className="font-serif text-2xl">{data.scope === "self" ? "Your month" : "Monthly rate"}</h2>
              <div className="mt-4 h-72">
                {data.scope === "self" ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data.personal_monthly}>
                      <CartesianGrid stroke="#E5DCD0" vertical={false} />
                      <XAxis dataKey="month" />
                      <YAxis allowDecimals={false} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="present" stackId="a" fill="#1F6B5A" />
                      <Bar dataKey="remote" stackId="a" fill="#7BA396" />
                      <Bar dataKey="leave" stackId="a" fill="#C4A574" />
                      <Bar dataKey="unplanned" stackId="a" fill="#9C5B3C" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={monthly}>
                      <CartesianGrid stroke="#E5DCD0" vertical={false} />
                      <XAxis dataKey="month" />
                      <YAxis unit="%" />
                      <Tooltip />
                      <Legend />
                      {Object.keys(monthly[0] || {}).filter((key) => key !== "month").slice(0, 4).map((key, index) => (
                        <Line key={key} type="monotone" dataKey={key} stroke={["#9C5B3C", "#1F6B5A", "#142233", "#C4A574"][index]} dot={false} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </Card>
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
            <Card className="min-w-0 overflow-hidden p-5">
              <h2 className="font-serif text-2xl">Anomalies</h2>
              <div className="mt-3 space-y-3">
                {data.anomalies.length === 0 ? <p className="text-sm text-ink/60">No department-month crossed the 1.5 z-score line.</p> : null}
                {data.anomalies.map((item) => (
                  <div key={`${item.department}-${item.month}`} className="rounded-xl bg-sand p-3 text-sm">
                    <div className="flex items-center justify-between">
                      <p className="font-semibold">{item.department}</p>
                      <Badge tone="copper">z {item.zscore}</Badge>
                    </div>
                    <p className="mt-1 text-ink/70">{item.month} · {pct(item.unplanned_absence_rate)}. {item.note}</p>
                  </div>
                ))}
              </div>
            </Card>
            <Card className="min-w-0 overflow-hidden p-5">
              <h2 className="font-serif text-2xl">What the numbers support</h2>
              <ul className="mt-3 space-y-3 text-sm leading-6">
                {data.insights.map((insight) => <li key={insight}>{insight}</li>)}
              </ul>
              <p className="mt-4 text-xs leading-5 text-ink/50">{data.method}</p>
            </Card>
          </div>
        </>
      ) : null}
    </div>
  );
}

function pivotMonthly(rows: Analytics["monthly"]) {
  const byMonth: Record<string, Record<string, string | number>> = {};
  for (const row of rows) {
    byMonth[row.month] = byMonth[row.month] || { month: row.month };
    byMonth[row.month][row.department] = Math.round(row.unplanned_absence_rate * 1000) / 10;
  }
  return Object.values(byMonth);
}
