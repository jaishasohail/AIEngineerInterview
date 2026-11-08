"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { API_URL, type SessionUser } from "@/lib/api";
import { Button, ErrorNote, Input } from "@/components/ui";

const DEMOS = [
  { name: "Aisha Rahman", role: "Employee", email: "aisha.rahman@northstar.example", note: "Leave planning and policy questions" },
  { name: "James Okonkwo", role: "Manager", email: "james.okonkwo@northstar.example", note: "Team overlap and approvals" },
  { name: "Helena Vogel", role: "HR admin", email: "helena.vogel@northstar.example", note: "Absence analytics and corrections" },
  { name: "Liam Chen", role: "Employee", email: "liam.chen@northstar.example", note: "Stale approval routing" },
];

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState(DEMOS[0].email);
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(nextEmail = email) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: nextEmail, password }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Sign-in failed");
      localStorage.setItem("helios_token", body.access_token);
      localStorage.setItem("helios_user", JSON.stringify(body.user as SessionUser));
      router.replace("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-screen lg:grid-cols-[1.1fr_0.9fr]">
      <section className="relative hidden overflow-hidden bg-ink text-sand lg:block">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(156,91,60,0.45),transparent_35%)]" />
        <div className="relative flex h-full flex-col justify-between p-12">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-sand/60">Northstar Global</p>
            <h1 className="mt-6 max-w-xl font-serif text-6xl leading-[0.95]">Helios decides with the record, then asks a human.</h1>
          </div>
          <div className="grid max-w-xl gap-4 text-sm leading-6 text-sand/80">
            <p>Leave plans, staffing conflicts, absence trends, and reporting fixes run through a LangGraph workflow. Numbers come from the simulated HRIS. The model does not approve anything.</p>
            <p>Organization clock is fixed at 7 Nov 2026 so the December Atlas deadline stays in the demo.</p>
          </div>
        </div>
      </section>
      <section className="flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-md">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-copper">Sign in</p>
          <h2 className="mt-2 font-serif text-4xl">HR Intelligence Copilot</h2>
          <form
            className="mt-8 space-y-4"
            onSubmit={(event) => {
              event.preventDefault();
              void submit();
            }}
          >
            <label className="block text-sm font-semibold">
              Work email
              <Input className="mt-1.5" value={email} onChange={(event) => setEmail(event.target.value)} type="email" required />
            </label>
            <label className="block text-sm font-semibold">
              Password
              <Input className="mt-1.5" value={password} onChange={(event) => setPassword(event.target.value)} type="password" required />
            </label>
            {error ? <ErrorNote message={error} /> : null}
            <Button className="w-full" disabled={busy} type="submit">
              {busy ? "Checking the directory…" : "Enter Helios"}
            </Button>
          </form>
          <div className="mt-8 space-y-2">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-ink/50">Demo accounts</p>
            <p className="text-xs leading-5 text-ink/60">Choosing an account fills the email only. Type the shared demo password from the project README. It is not included in this page.</p>
            {DEMOS.map((demo) => (
              <button
                key={demo.email}
                className="flex w-full items-center justify-between rounded-2xl border border-line bg-paper px-4 py-3 text-left hover:border-copper"
                onClick={() => {
                  setEmail(demo.email);
                }}
              >
                <span>
                  <span className="block text-sm font-semibold">{demo.name}</span>
                  <span className="block text-xs text-ink/60">{demo.note}</span>
                </span>
                <span className="text-xs font-semibold text-copper">{demo.role}</span>
              </button>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
