"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { CalendarRange, BarChart3, BookOpen, CalendarDays, CheckCheck, LayoutDashboard, LogOut, ScrollText, Settings, Sparkles, Users } from "lucide-react";
import { api, readUser, type SessionUser } from "@/lib/api";
import { cn } from "@/lib/api";

const NAV = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/assistant", label: "AI Assistant", icon: Sparkles },
  { href: "/leave", label: "Leave", icon: CalendarDays },
  { href: "/team", label: "Team", icon: Users },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/policies", label: "Policies", icon: BookOpen },
  { href: "/holidays", label: "Holidays", icon: CalendarRange },
  { href: "/approvals", label: "Approvals", icon: CheckCheck },
  { href: "/audit", label: "Audit log", icon: ScrollText },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Frame({ children }: { children: ReactNode }) {
  const path = usePathname();
  if (path === "/login" || path === "/") return <>{children}</>;
  return <Shell>{children}</Shell>;
}

function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<SessionUser | null>(null);

  useEffect(() => {
    const token = localStorage.getItem("helios_token");
    if (!token) {
      router.replace("/login");
      return;
    }
    setUser(readUser());
    api<SessionUser>("/api/auth/me")
      .then((profile) => {
        localStorage.setItem("helios_user", JSON.stringify(profile));
        setUser(profile);
      })
      .catch(() => undefined);
  }, [router]);

  const assistant = path === "/assistant";

  return (
    <div className="flex h-dvh flex-col overflow-hidden lg:grid lg:grid-cols-[260px_minmax(0,1fr)] lg:grid-rows-[minmax(0,1fr)]">
      <aside className="flex shrink-0 flex-col bg-ink text-sand lg:h-full lg:min-h-0 lg:overflow-hidden">
        <div className="shrink-0 px-5 py-6">
          <div className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-2xl bg-copper font-serif text-lg text-white">H</span>
            <div>
              <p className="font-serif text-xl leading-none">Helios</p>
              <p className="mt-1 text-[11px] uppercase tracking-[0.16em] text-sand/60">Northstar HR</p>
            </div>
          </div>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 lg:block lg:min-h-0 lg:flex-1 lg:space-y-1 lg:overflow-y-auto lg:overflow-x-hidden">
          {NAV.map((item) => {
            const active = path === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex shrink-0 items-center gap-3 rounded-xl px-3 py-2.5 text-sm",
                  active ? "bg-white/10 text-white" : "text-sand/75 hover:bg-white/5 hover:text-white",
                )}
              >
                <Icon size={16} />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="shrink-0 border-t border-white/10 px-5 py-4 lg:mt-auto">
          <p className="text-sm font-semibold">{user?.full_name || "Signed in"}</p>
          <p className="text-xs text-sand/60">{user ? `${user.role.replaceAll("_", " ")} · ${user.department}` : ""}</p>
          <button
            className="mt-3 flex items-center gap-2 text-xs text-sand/70 hover:text-white"
            onClick={() => {
              localStorage.removeItem("helios_token");
              localStorage.removeItem("helios_user");
              router.replace("/login");
            }}
          >
            <LogOut size={14} /> Sign out
          </button>
        </div>
      </aside>
      <main
        className={cn(
          "flex min-h-0 min-w-0 flex-1 flex-col px-4 py-4 sm:px-8 lg:px-10 lg:py-6",
          assistant ? "overflow-hidden" : "overflow-y-auto",
        )}
      >
        {children}
      </main>
    </div>
  );
}
