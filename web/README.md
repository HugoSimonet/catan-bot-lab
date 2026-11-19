# Catan Bot Lab: website

A fully static Next.js site (`output: "export"`) that presents the project's results and replays.
Everything it shows is pre-computed by the Python pipeline, so there is no backend.

```bash
npm install
npm run dev      # http://localhost:3000
npm run build    # static site in ./out
```

## Data

| File | Produced by |
|---|---|
| `data/summary.json` | `python -m catanbot.pipeline export` (tournament, tuning, sweeps, training curve, evaluation, style) |
| `public/data/replays/*.json` | `python -m catanbot.pipeline replays [--checkpoint ...]` |

Replays are complete game records: the board, the topology needed to draw it, and one frame per action.
Open any replay at a given action with `#t=<action>`, for example `/replays/ows-win/#t=200`.

## Deploying to Vercel

1. Push the repository to GitHub and import it in Vercel.
2. Set **Root Directory** to `web`. The framework preset (Next.js) and `vercel.json` handle the rest.
3. Optional: set `NEXT_PUBLIC_SITE_URL` to the production URL so Open Graph links are absolute.

`vercel.json` skips deployments for commits that do not touch `web/` (changes to the Rust engine or the
Python code don't change the site until the data is re-exported).

## Reusing the components in a portfolio site

The components have no dependencies beyond React:

- `components/HexBoard.tsx`: pure SVG board renderer (server- or client-side)
- `components/ReplayPlayer.tsx`: client component that loads `/data/replays/{id}.json` and plays it
- `components/charts.tsx`: `WinRateBars`, `LineChart`, `Legend` (pure SVG)

They use a few CSS variables and utility classes from `app/globals.css` (`--bg`, `--fg`, `--muted`, `--card`,
`--line`, `--accent`, `.card`, `.chip`). Copy those along with the components, plus the JSON files
from `public/data/` and `data/`.
