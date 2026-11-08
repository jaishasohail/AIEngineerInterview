"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function HomePage() {
  const router = useRouter();
  useEffect(() => {
    router.replace(localStorage.getItem("helios_token") ? "/dashboard" : "/login");
  }, [router]);
  return <p className="p-10 text-sm text-ink/60">Opening Helios…</p>;
}
