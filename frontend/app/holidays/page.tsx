"use client";

import { useEffect, useState } from "react";
import { api, readUser } from "@/lib/api";
import { Badge, Button, Card, ErrorNote, Field, Input, PageHeader } from "@/components/ui";

type Holiday = {
  id: number;
  name: string;
  holiday_date: string;
  country: string;
  region: string;
  department: string | null;
  is_active: boolean;
  source: string;
};

export default function HolidaysPage() {
  const [rows, setRows] = useState<Holiday[]>([]);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [role, setRole] = useState("");
  const [country, setCountry] = useState("");

  function load(nextCountry = country) {
    const query = nextCountry ? `?country=${encodeURIComponent(nextCountry)}` : "";
    api<{ holidays: Holiday[] }>(`/api/holidays${query}`)
      .then((body) => setRows(body.holidays))
      .catch((err: Error) => setError(err.message))
      .finally(() => setReady(true));
  }

  useEffect(() => {
    setRole(readUser()?.role || "");
    load("");
    // The first load is intentionally unfiltered.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <PageHeader
        kicker="Calendars"
        title="Holiday calendars"
        text="Leave math uses the office calendar for Austin, London, Berlin, and Singapore. An office with no rows keeps weekends only. Dates are never copied from another country."
      />
      {error ? <div className="mb-4"><ErrorNote message={error} /></div> : null}
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <Field label="Country">
          <Input value={country} placeholder="US, GB, DE, SG" onChange={(event) => setCountry(event.target.value.toUpperCase())} />
        </Field>
        <Button variant="ghost" onClick={() => load(country)}>Filter</Button>
        <Button variant="ghost" onClick={() => { setCountry(""); load(""); }}>All offices</Button>
      </div>
      {role === "hr_admin" ? (
        <Card className="mb-4 p-5">
          <h2 className="font-serif text-2xl">Add a holiday</h2>
          <form
            className="mt-4 grid gap-3 md:grid-cols-4"
            onSubmit={(event) => {
              event.preventDefault();
              const form = event.currentTarget;
              const data = new FormData(form);
              api<Holiday>("/api/holidays", {
                method: "POST",
                body: JSON.stringify({
                  name: data.get("name"),
                  holiday_date: data.get("holiday_date"),
                  country: data.get("country"),
                  region: data.get("region") || "",
                  is_active: true,
                }),
              })
                .then(() => {
                  form.reset();
                  load(country);
                })
                .catch((err: Error) => setError(err.message));
            }}
          >
            <Field label="Name"><Input name="name" required /></Field>
            <Field label="Date"><Input name="holiday_date" type="date" required /></Field>
            <Field label="Country"><Input name="country" required placeholder="US" /></Field>
            <Field label="Office"><Input name="region" placeholder="Austin, or blank for the country" /></Field>
            <Button className="md:col-span-4 w-fit" type="submit">Save holiday</Button>
          </form>
        </Card>
      ) : null}
      <Card className="overflow-hidden">
        {!ready && !error ? <div className="p-5 text-sm text-ink/60">Loading calendars…</div> : null}
        {ready ? <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="bg-sand/70 text-xs uppercase tracking-wide text-ink/60">
            <tr>
              <th className="px-4 py-3">Date</th>
              <th className="px-4 py-3">Name</th>
              <th className="px-4 py-3">Country</th>
              <th className="px-4 py-3">Office</th>
              <th className="px-4 py-3">Status</th>
              {role === "hr_admin" ? <th className="px-4 py-3" /> : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className="border-t border-line">
                <td className="px-4 py-3">{row.holiday_date}</td>
                <td className="px-4 py-3 font-semibold">{row.name}</td>
                <td className="px-4 py-3">{row.country}</td>
                <td className="px-4 py-3">{row.region || row.department || "All offices"}</td>
                <td className="px-4 py-3"><Badge tone={row.is_active ? "pine" : "warn"}>{row.is_active ? row.source : "inactive"}</Badge></td>
                {role === "hr_admin" ? (
                  <td className="px-4 py-3 text-right">
                    <button
                      className="mr-3 text-xs font-semibold text-copper"
                      onClick={() => {
                        api(`/api/holidays/${row.id}`, { method: "PATCH", body: JSON.stringify({ is_active: !row.is_active }) })
                          .then(() => load(country))
                          .catch((err: Error) => setError(err.message));
                      }}
                    >
                      {row.is_active ? "Deactivate" : "Activate"}
                    </button>
                    <button
                      className="text-xs font-semibold text-[#9B2C2C]"
                      onClick={() => {
                        api(`/api/holidays/${row.id}`, { method: "DELETE" })
                          .then(() => load(country))
                          .catch((err: Error) => setError(err.message));
                      }}
                    >
                      Delete
                    </button>
                  </td>
                ) : null}
              </tr>
            ))}
          </tbody>
        </table>
        </div> : null}
      </Card>
    </div>
  );
}
