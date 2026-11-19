import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import ReplayPlayer from "@/components/ReplayPlayer";
import { REPLAYS } from "@/lib/data";
import { botLabel } from "@/lib/theme";

export const dynamicParams = false;

export function generateStaticParams() {
  return REPLAYS.map((r) => ({ id: r.id }));
}

export async function generateMetadata(props: PageProps<"/replays/[id]">): Promise<Metadata> {
  const { id } = await props.params;
  const meta = REPLAYS.find((r) => r.id === id);
  return { title: meta ? meta.title : "Replay" };
}

export default async function ReplayPage(props: PageProps<"/replays/[id]">) {
  const { id } = await props.params;
  const meta = REPLAYS.find((r) => r.id === id);
  if (!meta) notFound();
  return (
    <div>
      <Link href="/replays/" className="link text-sm">← all replays</Link>
      <h1 className="mt-3 font-display text-3xl font-semibold">{meta.title}</h1>
      <p className="mt-2 text-muted">
        {meta.players.map(botLabel).join(" · ")}
        {meta.description ? `: ${meta.description}` : ""}
      </p>
      <div className="mt-6">
        <ReplayPlayer id={meta.id} />
      </div>
    </div>
  );
}
