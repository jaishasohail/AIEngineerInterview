"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Badge, Card, Empty, ErrorNote, PageHeader, statusTone } from "@/components/ui";

type Member = {
  id: number;
  full_name: string;
  job_title: string;
  department: string;
  location: string;
  manager: string | null;
  leave: { start_date: string; end_date: string; status: string; business_days: number; reason: string | null }[];
  projects: { name: string; deadline: string; criticality: string; allocation_pct: number }[];
};

export default function TeamPage() {
  const [members, setMembers] = useState<Member[]>([]);
  const [scope, setScope] = useState("");
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    api<{ scope: string; members: Member[] }>("/api/team")
      .then((body) => {
        setScope(body.scope);
        setMembers(body.members);
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setReady(true));
  }, []);

  return (
    <div>
      <PageHeader
        kicker="Team"
        title="People and coverage"
        text={scope === "employee" ? "You see teammates' away dates, not their reasons or pay." : "Managers see direct reports. HR sees the directory slice returned for this screen."}
      />
      {error ? <ErrorNote message={error} /> : null}
      {ready && members.length === 0 && !error ? <Empty>No teammates were returned for this account.</Empty> : null}
      {!ready && !error ? <p className="text-sm text-ink/60">Loading the team…</p> : null}
      <div className="grid gap-4 lg:grid-cols-2">
        {members.map((member) => (
          <Card key={member.id} className="p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="font-serif text-2xl">{member.full_name}</h2>
                <p className="text-sm text-ink/70">{member.job_title} · {member.department} · {member.location}</p>
              </div>
              {member.manager ? <Badge>Reports to {member.manager}</Badge> : null}
            </div>
            <div className="mt-4">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-ink/45">Projects</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {member.projects.length === 0 ? <span className="text-sm text-ink/50">No active assignment</span> : null}
                {member.projects.map((project) => (
                  <Badge key={project.name} tone={project.criticality === "critical" ? "bad" : "pine"}>
                    {project.name} · {project.deadline}
                  </Badge>
                ))}
              </div>
            </div>
            <div className="mt-4 space-y-2">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-ink/45">Leave</p>
              {member.leave.length === 0 ? <p className="text-sm text-ink/50">No upcoming leave.</p> : null}
              {member.leave.map((leave) => (
                <div key={`${leave.start_date}-${leave.status}`} className="flex items-center justify-between rounded-xl bg-sand px-3 py-2 text-sm">
                  <span>{leave.start_date} → {leave.end_date} · {leave.business_days} days {leave.reason ? `· ${leave.reason}` : ""}</span>
                  <Badge tone={statusTone(leave.status)}>{leave.status.replaceAll("_", " ")}</Badge>
                </div>
              ))}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
