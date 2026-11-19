"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import HexBoard from "@/components/HexBoard";
import { actionTarget, buildSnapshots, turnStarts } from "@/lib/replay";
import {
  PLAYER_COLORS,
  PLAYER_STROKE,
  RESOURCE_FILL,
  RESOURCE_LABELS,
  botLabel,
} from "@/lib/theme";
import type { Frame, Replay } from "@/lib/types";

const SPEEDS = [
  { label: "1×", fps: 3 },
  { label: "2×", fps: 6 },
  { label: "4×", fps: 12 },
  { label: "8×", fps: 30 },
];

interface Props {
  /** Replay id: loads /data/replays/{id}.json */
  id: string;
  autoplay?: boolean;
  compact?: boolean;
}

export default function ReplayPlayer({ id, autoplay = false, compact = false }: Props) {
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/data/replays/${id}.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: Replay) => !cancelled && setReplay(data))
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (error) return <div className="card p-6 text-sm">Could not load replay: {error}</div>;
  if (!replay) return <div className="card p-6 text-sm text-muted animate-pulse">Loading replay…</div>;
  return <Player replay={replay} autoplay={autoplay} compact={compact} />;
}

function Player({ replay, autoplay, compact }: { replay: Replay; autoplay: boolean; compact: boolean }) {
  const n = replay.frames.length;
  const snapshots = useMemo(() => buildSnapshots(replay), [replay]);
  const starts = useMemo(() => turnStarts(replay), [replay]);
  // Deep link: #t=123 opens the replay at action 123. (This component only
  // mounts client-side, after the replay is fetched, so `window` is available.)
  const linked = compact ? null : hashFrame(n);
  const [k, setK] = useState(linked ?? 0); // frames applied
  const [playing, setPlaying] = useState(autoplay && linked === null);
  const [speed, setSpeed] = useState(autoplay ? 2 : 1);
  const logRef = useRef<HTMLOListElement>(null);

  const frame: Frame | null = k > 0 ? replay.frames[k - 1] : null;
  let lastRoll: number | null = null;
  for (let i = k - 1; i >= 0; i--) {
    const r = replay.frames[i].roll;
    if (r !== null) {
      lastRoll = r;
      break;
    }
  }

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setK((x) => {
        if (x >= n) {
          setPlaying(false);
          return x;
        }
        return x + 1;
      });
    }, 1000 / SPEEDS[speed].fps);
    return () => clearInterval(id);
  }, [playing, speed, n]);

  const jumpTurn = useCallback(
    (dir: 1 | -1) => {
      setK((x) => {
        const cur = x - 1;
        if (dir > 0) {
          const next = starts.find((s) => s > cur);
          return next === undefined ? n : next + 1;
        }
        const prev = [...starts].reverse().find((s) => s < cur - 1);
        return prev === undefined ? 0 : prev + 1;
      });
    },
    [starts, n],
  );

  useEffect(() => {
    if (compact) return;
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        if (e.shiftKey) jumpTurn(1);
        else setK((x) => Math.min(n, x + 1));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        if (e.shiftKey) jumpTurn(-1);
        else setK((x) => Math.max(0, x - 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [compact, jumpTurn, n]);

  useEffect(() => {
    const el = logRef.current?.querySelector('[data-current="true"]');
    el?.scrollIntoView({ block: "nearest" });
  }, [k]);

  const highlight = frame ? { ...actionTarget(frame.action), player: frame.actor } : undefined;
  const robber = frame ? frame.robber : replay.robber_start;
  const finished = k >= n;

  const board = (
    <HexBoard
      topology={replay.topology}
      board={replay.board}
      pieces={snapshots[k]}
      robber={robber}
      highlight={highlight}
      className="w-full h-auto"
    />
  );

  if (compact) {
    return (
      <div>
        {board}
        <div className="mt-1 grid grid-cols-2 gap-1.5 px-1 text-xs">
          {replay.players.map((p, i) => (
            <span key={i} className={`chip justify-between ${frame?.actor === i ? "ring-1 ring-accent" : ""}`}>
              <span className="inline-flex items-center gap-1.5 truncate">
                <span className="dot shrink-0" style={{ background: PLAYER_COLORS[i], borderColor: PLAYER_STROKE[i] }} />
                {botLabel(p)}
                {finished && replay.winner === i && <Trophy />}
              </span>
              <span className="font-semibold tabular-nums">{frame ? frame.vp[i] : 0} VP</span>
            </span>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div className="card p-3 sm:p-4">
        <div className="relative">
          {board}
          <div className="absolute right-2 top-2 text-right">
            <div className="text-[11px] uppercase tracking-wider text-white/80">
              Turn {frame ? frame.turn : 0} / {replay.turns}
            </div>
            {lastRoll !== null && (
              <div className="mt-1 inline-flex items-center gap-1.5 rounded-md bg-black/45 px-2 py-1 font-mono text-sm text-white">
                <DiceIcon /> {lastRoll}
              </div>
            )}
          </div>
          {frame && (
            <div className="absolute left-2 bottom-2 max-w-[80%] rounded-md bg-black/50 px-2.5 py-1.5 text-sm text-white">
              <span className="font-semibold" style={{ color: PLAYER_COLORS[frame.actor] }}>
                P{frame.actor + 1}
              </span>{" "}
              {frame.text}
            </div>
          )}
        </div>

        <Controls
          k={k}
          n={n}
          playing={playing}
          speed={speed}
          onPlay={() => {
            if (k >= n) setK(0);
            setPlaying((p) => !p);
          }}
          onSpeed={setSpeed}
          onSeek={(v) => {
            setPlaying(false);
            setK(v);
          }}
          onStep={(d) => {
            setPlaying(false);
            setK((x) => Math.max(0, Math.min(n, x + d)));
          }}
          onTurn={(d) => {
            setPlaying(false);
            jumpTurn(d);
          }}
        />
      </div>

      <aside className="flex flex-col gap-3">
        {replay.players.map((p, i) => (
          <PlayerCard key={i} seat={i} name={p} frame={frame} active={frame?.actor === i}
            winner={finished && replay.winner === i} />
        ))}
        <div className="card p-3">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">Game log</div>
          <ol ref={logRef} className="max-h-56 overflow-y-auto pr-1 text-[13px] leading-snug">
            {replay.frames.slice(Math.max(0, k - 60), k).map((f, j) => {
              const idx = Math.max(0, k - 60) + j;
              const current = idx === k - 1;
              return (
                <li key={idx} data-current={current}
                  className={`flex cursor-pointer gap-2 rounded px-1 py-0.5 ${current ? "bg-accent/15" : "hover:bg-black/5 dark:hover:bg-white/5"}`}
                  onClick={() => { setPlaying(false); setK(idx + 1); }}>
                  <span className="w-7 shrink-0 text-right font-mono text-[11px] text-muted">{f.turn}</span>
                  <span className="font-semibold" style={{ color: PLAYER_COLORS[f.actor] === "#f2efe6" ? "inherit" : PLAYER_COLORS[f.actor] }}>
                    P{f.actor + 1}
                  </span>
                  <span>{f.text}</span>
                </li>
              );
            })}
            {k === 0 && <li className="text-muted">Press play (or space) to start.</li>}
          </ol>
        </div>
        <p className="text-xs text-muted">
          Keys: <kbd>space</kbd> play/pause · <kbd>←</kbd>/<kbd>→</kbd> step · <kbd>shift</kbd>+arrows jump a turn.
          Replays show every player&apos;s hand; the bots themselves only saw card counts.
        </p>
      </aside>
    </div>
  );
}

function Controls(props: {
  k: number;
  n: number;
  playing: boolean;
  speed: number;
  onPlay: () => void;
  onSpeed: (s: number) => void;
  onSeek: (v: number) => void;
  onStep: (d: number) => void;
  onTurn: (d: 1 | -1) => void;
}) {
  const { k, n, playing, speed } = props;
  const btn = "rounded-md border border-line px-2.5 py-1.5 text-sm hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-40";
  return (
    <div className="mt-3 flex flex-col gap-2">
      <input
        type="range"
        min={0}
        max={n}
        value={k}
        onChange={(e) => props.onSeek(Number(e.target.value))}
        className="w-full accent-[var(--accent)]"
        aria-label="Seek"
      />
      <div className="flex flex-wrap items-center gap-1.5">
        <button className={btn} onClick={() => props.onSeek(0)} aria-label="Start" title="Start">
          <Icon d="M6 5v14M19 5l-9 7 9 7z" />
        </button>
        <button className={btn} onClick={() => props.onTurn(-1)} aria-label="Previous turn" title="Previous turn (shift+←)">
          <Icon d="M12 5l-8 7 8 7zM21 5l-8 7 8 7z" />
        </button>
        <button className={btn} onClick={() => props.onStep(-1)} aria-label="Step back" title="Step back (←)">
          <Icon d="M16 5l-9 7 9 7z" />
        </button>
        <button className={`${btn} min-w-20 font-semibold`} onClick={props.onPlay}>
          {playing ? "Pause" : k >= n ? "Replay" : "Play"}
        </button>
        <button className={btn} onClick={() => props.onStep(1)} aria-label="Step forward" title="Step forward (→)">
          <Icon d="M8 5l9 7-9 7z" />
        </button>
        <button className={btn} onClick={() => props.onTurn(1)} aria-label="Next turn" title="Next turn (shift+→)">
          <Icon d="M3 5l8 7-8 7zM12 5l8 7-8 7z" />
        </button>
        <button className={btn} onClick={() => props.onSeek(n)} aria-label="End" title="End">
          <Icon d="M18 5v14M5 5l9 7-9 7z" />
        </button>
        <div className="ml-auto flex items-center gap-1">
          {SPEEDS.map((s, i) => (
            <button key={s.label} onClick={() => props.onSpeed(i)}
              className={`${btn} ${i === speed ? "bg-accent text-white hover:bg-accent" : ""}`}>
              {s.label}
            </button>
          ))}
        </div>
      </div>
      <div className="font-mono text-xs text-muted">
        action {k} / {n}
      </div>
    </div>
  );
}

function hashFrame(n: number): number | null {
  if (typeof window === "undefined") return null;
  const m = window.location.hash.match(/t=(\d+)/);
  return m ? Math.min(n, Number(m[1])) : null;
}

function Icon({ d }: { d: string }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2"
      strokeLinejoin="round" aria-hidden className="block">
      <path d={d} />
    </svg>
  );
}

function DiceIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden>
      <rect x="3" y="3" width="18" height="18" rx="4" fill="none" stroke="currentColor" strokeWidth="2" />
      <circle cx="8.5" cy="8.5" r="1.6" fill="currentColor" />
      <circle cx="15.5" cy="15.5" r="1.6" fill="currentColor" />
      <circle cx="12" cy="12" r="1.6" fill="currentColor" />
    </svg>
  );
}

function Trophy() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" aria-label="Winner" role="img" className="inline-block text-[#e3b448]">
      <path fill="currentColor" d="M7 3h10v3h3v2a5 5 0 0 1-4.6 5A5 5 0 0 1 13 15.9V18h3v3H8v-3h3v-2.1A5 5 0 0 1 8.6 13 5 5 0 0 1 4 8V6h3V3zm-1 5a3 3 0 0 0 1.4 2.5A7 7 0 0 1 7 8zm12 0h-1a7 7 0 0 1-.4 2.5A3 3 0 0 0 18 8z" />
    </svg>
  );
}

function PlayerCard({ seat, name, frame, active, winner }: {
  seat: number;
  name: string;
  frame: Frame | null;
  active: boolean;
  winner: boolean;
}) {
  const hand = frame ? frame.hands[seat] : [0, 0, 0, 0, 0];
  const devs = frame ? frame.dev_cards[seat].reduce((a, b) => a + b, 0) : 0;
  const vp = frame ? frame.vp[seat] : 0;
  return (
    <div
      className={`card p-3 transition ${active ? "ring-2" : ""}`}
      style={active ? ({ "--tw-ring-color": PLAYER_COLORS[seat] } as React.CSSProperties) : undefined}
    >
      <div className="flex items-center gap-2">
        <span className="h-3.5 w-3.5 rounded-full border-2"
          style={{ background: PLAYER_COLORS[seat], borderColor: PLAYER_STROKE[seat] }} />
        <span className="font-semibold">P{seat + 1}</span>
        <span className="truncate text-sm text-muted">{botLabel(name)}</span>
        <span className="ml-auto font-display text-2xl font-semibold tabular-nums">
          {vp}
          <span className="ml-0.5 text-xs font-normal text-muted">VP</span>
        </span>
        {winner && <Trophy />}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        {hand.map((c, r) => (
          <span key={r} title={RESOURCE_LABELS[r]}
            className="inline-flex min-w-9 items-center justify-center gap-1 rounded px-1.5 py-0.5 font-mono text-xs text-white"
            style={{ background: RESOURCE_FILL[r], opacity: c ? 1 : 0.35 }}>
            {c}
          </span>
        ))}
        <span className="ml-1 text-xs text-muted" title="Development cards held">+{devs} dev</span>
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5 text-[11px]">
        <span className="chip">Knights {frame ? frame.knights[seat] : 0}</span>
        <span className="chip">Road {frame ? frame.road_length[seat] : 0}</span>
        {frame?.longest_road === seat && <span className="chip chip-accent">Longest Road</span>}
        {frame?.largest_army === seat && <span className="chip chip-accent">Largest Army</span>}
      </div>
    </div>
  );
}
