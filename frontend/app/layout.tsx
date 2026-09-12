import type { Metadata } from "next";
import type { ReactNode } from "react";

import { AppShell } from "@/components/app-shell";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "MLForge",
    template: "%s | MLForge",
  },
  description: "A precise local workspace for reproducible tabular machine learning.",
};

type RootLayoutProps = Readonly<{
  children: ReactNode;
}>;

const themeInitialization = `
try {
  const saved = localStorage.getItem("mlforge-theme");
  document.documentElement.dataset.theme = saved === "light" || saved === "dark" ? saved : "system";
} catch {
  document.documentElement.dataset.theme = "system";
}
`;

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" data-theme="system" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitialization }} />
      </head>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
