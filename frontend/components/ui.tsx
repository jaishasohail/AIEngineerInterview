import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, TextareaHTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/api";

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <section className={cn("rounded-2xl border border-line bg-paper shadow-card", className)}>{children}</section>;
}

const buttonStyles = cva(
  "inline-flex items-center justify-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-50",
  {
    variants: {
      variant: {
        primary: "bg-ink text-sand hover:bg-ink/90",
        copper: "bg-copper text-white hover:bg-copper/90",
        ghost: "border border-line bg-white text-ink hover:bg-sand",
        danger: "bg-[#9B2C2C] text-white hover:bg-[#7f2424]",
      },
    },
    defaultVariants: { variant: "primary" },
  },
);

export function Button({
  className,
  variant,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonStyles>) {
  return <button className={cn(buttonStyles({ variant }), className)} {...props} />;
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-sm">
      <span className="mb-1.5 block font-semibold text-ink/80">{label}</span>
      {children}
    </label>
  );
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "w-full rounded-xl border border-line bg-white px-3 py-2.5 text-sm outline-none ring-copper/30 focus:ring-2",
        props.className,
      )}
    />
  );
}

export function TextArea(props: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={cn(
        "w-full rounded-xl border border-line bg-white px-3 py-2.5 text-sm outline-none ring-copper/30 focus:ring-2",
        props.className,
      )}
    />
  );
}

export function Badge({ children, tone = "ink" }: { children: ReactNode; tone?: "ink" | "pine" | "copper" | "warn" | "bad" }) {
  const tones = {
    ink: "bg-ink/5 text-ink",
    pine: "bg-pine/10 text-pine",
    copper: "bg-copper/10 text-copper",
    warn: "bg-amber-100 text-amber-800",
    bad: "bg-red-100 text-red-800",
  };
  return <span className={cn("rounded-full px-2.5 py-1 text-xs font-semibold", tones[tone])}>{children}</span>;
}

export function statusTone(status: string): "ink" | "pine" | "copper" | "warn" | "bad" {
  if (status === "approved" || status === "allowed" || status === "success") return "pine";
  if (status === "rejected" || status === "denied" || status === "blocked") return "bad";
  if (status === "pending_hr" || status === "conflict") return "copper";
  if (status.includes("pending") || status === "high") return "warn";
  return "ink";
}

export function PageHeader({ kicker, title, text }: { kicker: string; title: string; text: string }) {
  return (
    <header className="mb-6">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-copper">{kicker}</p>
      <h1 className="mt-1 font-serif text-4xl text-ink">{title}</h1>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-ink/70">{text}</p>
    </header>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="rounded-xl border border-dashed border-line px-4 py-8 text-center text-sm text-ink/60">{children}</p>;
}

export function ErrorNote({ message }: { message: string }) {
  return <p className="rounded-xl bg-red-50 px-3 py-2 text-sm text-red-800">{message}</p>;
}
