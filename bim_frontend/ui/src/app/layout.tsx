import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BIMClaw — BIM Spatial Understanding",
  description: "Answer spatial questions about IFC / BIM files using AI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
