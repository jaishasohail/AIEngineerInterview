"use client";

import { useEffect, useState } from "react";
import { api, apiForm, readUser } from "@/lib/api";
import { Badge, Button, Card, ErrorNote, PageHeader, statusTone } from "@/components/ui";

type Policy = {
  slug: string;
  title: string;
  version: string;
  status: string;
  effective_date: string;
  expiry_date: string | null;
  department: string | null;
  access: string;
  conflict_group: string | null;
  supersedes: string | null;
  index_status?: string;
  embedding_model?: string | null;
  chunk_count?: number;
  index_error?: string | null;
};

export default function PoliciesPage() {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [selected, setSelected] = useState<{ title: string; body: string; status: string; version: string } | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [role, setRole] = useState<string>("");
  const [busy, setBusy] = useState(false);

  function load() {
    api<{ policies: Policy[] }>("/api/policies").then((body) => setPolicies(body.policies)).catch((err: Error) => setError(err.message));
  }

  useEffect(() => {
    setRole(readUser()?.role || "");
    load();
  }, []);

  return (
    <div>
      <PageHeader
        kicker="Knowledge"
        title="Policy library"
        text="Current documents are embedded with the configured dense model and searched with BM25 plus reciprocal rank fusion. Expired versions stay in the library. Retrieval drops them when a current successor exists."
      />
      {error ? <ErrorNote message={error} /> : null}
      {notice ? <p className="mb-4 text-sm text-pine">{notice}</p> : null}
      {role === "hr_admin" ? (
        <Card className="mb-4 p-5">
          <h2 className="font-serif text-2xl">Index</h2>
          <p className="mt-1 text-sm text-ink/70">Upload a PDF, DOCX, TXT, or Markdown file. Reindex rewrites the vectors for the active embedding model.</p>
            <form
            className="mt-4 flex flex-col items-stretch gap-3 sm:flex-row sm:flex-wrap sm:items-end"
            onSubmit={(event) => {
              event.preventDefault();
              const form = event.currentTarget;
              const data = new FormData(form);
              setBusy(true);
              setError("");
              apiForm<Policy>("/api/policies/upload", data)
                .then((row) => {
                  setNotice(`${row.title} indexed with ${row.embedding_model || "the active model"}.`);
                  form.reset();
                  load();
                })
                .catch((err: Error) => setError(err.message))
                .finally(() => setBusy(false));
            }}
          >
            <label className="text-sm">
              <span className="mb-1 block font-semibold">File</span>
              <input name="file" type="file" required accept=".pdf,.docx,.txt,.md" className="text-sm" />
            </label>
            <label className="text-sm">
              <span className="mb-1 block font-semibold">Title</span>
              <input name="title" required className="rounded-xl border border-line px-3 py-2" />
            </label>
            <label className="text-sm">
              <span className="mb-1 block font-semibold">Version</span>
              <input name="version" required defaultValue="1.0" className="w-24 rounded-xl border border-line px-3 py-2" />
            </label>
            <label className="text-sm">
              <span className="mb-1 block font-semibold">Family</span>
              <input name="policy_family" required defaultValue="general" className="rounded-xl border border-line px-3 py-2" />
            </label>
            <label className="text-sm">
              <span className="mb-1 block font-semibold">Effective</span>
              <input name="effective_date" type="date" required defaultValue="2026-01-01" className="rounded-xl border border-line px-3 py-2" />
            </label>
            <Button disabled={busy} type="submit">{busy ? "Indexing…" : "Upload and index"}</Button>
            <Button
              type="button"
              variant="ghost"
              disabled={busy}
              onClick={() => {
                setBusy(true);
                setError("");
                api<{ results: { slug: string; outcome: string }[] }>("/api/policies/reindex", { method: "POST" })
                  .then((body) => {
                    const failed = body.results.filter((item) => item.outcome === "failed").length;
                    setNotice(failed ? `Reindex finished with ${failed} failure${failed === 1 ? "" : "s"}.` : "Reindex finished. Every current vector was regenerated.");
                    load();
                  })
                  .catch((err: Error) => setError(err.message))
                  .finally(() => setBusy(false));
              }}
            >
              Reindex all
            </Button>
          </form>
        </Card>
      ) : null}
      <div className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="space-y-2">
          {policies.map((policy) => (
            <div
              key={policy.slug}
              className="w-full rounded-2xl border border-line bg-paper px-4 py-3 text-left hover:border-copper"
              onClick={() => {
                api<{ title: string; body: string; status: string; version: string }>(`/api/policies/${policy.slug}`)
                  .then(setSelected)
                  .catch((err: Error) => setError(err.message));
              }}
            >
              <div className="flex items-center justify-between gap-3">
                <button
                  className="text-left font-semibold"
                  onClick={() => {
                    api<{ title: string; body: string; status: string; version: string }>(`/api/policies/${policy.slug}`)
                      .then(setSelected)
                      .catch((err: Error) => setError(err.message));
                  }}
                >
                  {policy.title}
                </button>
                <Badge tone={statusTone(policy.status)}>{policy.status}</Badge>
              </div>
              <p className="mt-1 text-xs text-ink/60">
                v{policy.version} · effective {policy.effective_date}
                {policy.index_status ? ` · index ${policy.index_status}` : ""}
                {policy.chunk_count !== undefined ? ` · ${policy.chunk_count} chunks` : ""}
                {policy.conflict_group ? ` · conflict ${policy.conflict_group}` : ""}
                {policy.supersedes ? ` · supersedes ${policy.supersedes}` : ""}
              </p>
              {role === "hr_admin" ? (
                <span className="mt-2 flex gap-2">
                  <button
                    className="text-xs font-semibold text-copper"
                    onClick={(event) => {
                      event.stopPropagation();
                      setBusy(true);
                      api(`/api/policies/${policy.slug}/reindex`, { method: "POST" })
                        .then(() => {
                          setNotice(`${policy.title} was reindexed.`);
                          load();
                        })
                        .catch((err: Error) => setError(err.message))
                        .finally(() => setBusy(false));
                    }}
                  >
                    Reindex
                  </button>
                  <button
                    className="text-xs font-semibold text-[#9B2C2C]"
                    onClick={(event) => {
                      event.stopPropagation();
                      setBusy(true);
                      api(`/api/policies/${policy.slug}`, { method: "DELETE" })
                        .then(() => {
                          setNotice(`${policy.title} was removed from the index.`);
                          if (selected?.title === policy.title) setSelected(null);
                          load();
                        })
                        .catch((err: Error) => setError(err.message))
                        .finally(() => setBusy(false));
                    }}
                  >
                    Remove
                  </button>
                </span>
              ) : null}
            </div>
          ))}
        </div>
        <Card className="p-5">
          {selected ? (
            <>
              <div className="flex items-center justify-between">
                <h2 className="font-serif text-3xl">{selected.title}</h2>
                <Badge tone={statusTone(selected.status)}>v{selected.version}</Badge>
              </div>
              <article className="prose-sm mt-4 whitespace-pre-wrap text-sm leading-6">{selected.body}</article>
            </>
          ) : <p className="text-sm text-ink/60">Select a document to read the ingested text.</p>}
        </Card>
      </div>
    </div>
  );
}
