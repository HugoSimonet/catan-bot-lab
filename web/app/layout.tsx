import type { Metadata } from "next";
import { Fraunces, Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";

import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });
const fraunces = Fraunces({ variable: "--font-fraunces", subsets: ["latin"] });

// Absolute base for link previews: an explicit URL if set, else Vercel's production domain (set at build time).
const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ??
  (process.env.VERCEL_PROJECT_PRODUCTION_URL
    ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
    : "http://localhost:3000");

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "Catan Bot Lab: strategy bots vs. a self-taught AI",
    template: "%s · Catan Bot Lab",
  },
  description:
    "Millions of simulated Settlers of Catan games: community strategies as bots, how small tweaks change their win rates, and a reinforcement-learning agent that learns its own way to win.",
  openGraph: {
    title: "Catan Bot Lab",
    description:
      "Community Catan strategies as bots, millions of simulated games, and a reinforcement-learning agent trained to beat them.",
    type: "website",
  },
  twitter: { card: "summary_large_image" },
};

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/results/", label: "Results" },
  { href: "/improving/", label: "Improvements" },
  { href: "/replays/", label: "Replays" },
  { href: "/how-it-works/", label: "How it works" },
];

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} ${fraunces.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <header className="sticky top-0 z-20 border-b border-line bg-bg/85 backdrop-blur">
          <nav className="mx-auto flex max-w-6xl items-center gap-1 px-4 py-3 sm:gap-4">
            <Link href="/" className="mr-2 flex items-center gap-2 font-display text-lg font-semibold">
              <HexMark />
              <span className="hidden sm:inline">Catan Bot Lab</span>
            </Link>
            {NAV.slice(1).map((n) => (
              <Link key={n.href} href={n.href} className="rounded-md px-2 py-1 text-sm text-muted hover:bg-black/5 hover:text-fg dark:hover:bg-white/10">
                {n.label}
              </Link>
            ))}
          </nav>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:py-12">{children}</main>
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-6 text-xs text-muted">
            <span>
              A personal research project. Not affiliated with or endorsed by the publishers of CATAN®.
            </span>
            <span>Rust engine · PyTorch PPO · Next.js</span>
          </div>
        </footer>
      </body>
    </html>
  );
}

function HexMark() {
  return (
    <svg width="26" height="26" viewBox="-1.1 -1.1 2.2 2.2" aria-hidden>
      <polygon points="0,-1 0.866,-0.5 0.866,0.5 0,1 -0.866,0.5 -0.866,-0.5" fill="#c8553d" />
      <polygon points="0,-0.55 0.476,-0.275 0.476,0.275 0,0.55 -0.476,0.275 -0.476,-0.275" fill="#e3b448" />
      <circle r="0.2" fill="#f6efdf" />
    </svg>
  );
}
