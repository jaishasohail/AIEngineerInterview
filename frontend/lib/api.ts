import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type SessionUser = {
  id: number;
  email: string;
  role: "employee" | "manager" | "hr_admin";
  employee_id: number;
  full_name: string;
  job_title: string;
  department: string | null;
  location: string;
};

export function readUser(): SessionUser | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem("helios_user");
  if (!raw) return null;
  try {
    return JSON.parse(raw) as SessionUser;
  } catch {
    return null;
  }
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  return readJson<T>(path, options, true);
}

export async function apiForm<T>(path: string, form: FormData): Promise<T> {
  return readJson<T>(path, { method: "POST", body: form }, false);
}

async function readJson<T>(path: string, options: RequestInit, json: boolean): Promise<T> {
  const token = typeof window === "undefined" ? null : localStorage.getItem("helios_token");
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      ...(json ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  });
  const body = await response.json().catch(() => ({}));
  if (response.status === 401 && typeof window !== "undefined" && !path.startsWith("/api/auth/login")) {
    localStorage.removeItem("helios_token");
    localStorage.removeItem("helios_user");
    window.location.href = "/login";
  }
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "The request could not be completed.";
    throw new Error(detail);
  }
  return body as T;
}

export function pct(value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value * 1000) / 10}%`;
}

export function titleCase(value: string) {
  return value.replaceAll("_", " ");
}
