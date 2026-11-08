import type { ReactNode } from "react";
import type { Metadata } from "next";
import { Fraunces, Source_Sans_3 } from "next/font/google";
import { Frame } from "@/components/Frame";
import "./globals.css";

const sans = Source_Sans_3({ subsets: ["latin"], variable: "--font-sans" });
const serif = Fraunces({ subsets: ["latin"], variable: "--font-serif" });

export const metadata: Metadata = {
  title: "Helios HR Intelligence Copilot",
  description: "Agentic HR operations for Northstar Global",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className={`${sans.variable} ${serif.variable} min-w-0 overflow-x-clip font-sans antialiased`}>
        <Frame>{children}</Frame>
      </body>
    </html>
  );
}
