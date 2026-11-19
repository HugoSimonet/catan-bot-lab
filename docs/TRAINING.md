# Training the agent on a Windows PC

How the published agent was trained (`configs/main.yaml`: 8,000 PPO updates, about 16 hours on a GTX 1660 Ti)
and how to reproduce it. Set up the Python environment first (see "Reproduce" in the README). Run every
command from the repository folder in PowerShell.

## 1. Before you start

- **Close memory-heavy apps** (browsers with many tabs, games). Training uses the GPU and all CPU cores.
- **Sleep:** set the PC to never sleep while plugged in (Settings → System → Power).
- **Windows Update:** pause updates for a day so an automatic restart can't interrupt the run.

## 2. Start training

A fresh run:

```powershell
.\.venv\Scripts\python.exe -m catanbot.train --config configs/main.yaml
```

To continue an interrupted run, and restart automatically (up to 5 times) if it crashes:

```powershell
$tries = 0; do { .\.venv\Scripts\python.exe -m catanbot.train --config configs/main.yaml --resume; $tries++ } while ($LASTEXITCODE -ne 0 -and $tries -lt 5)
```

One line prints per update, and every 50 updates an evaluation line appears:

```
upd  1201 | strategies | win 0.154 vp 6.46 turns 81 | ent 0.472 kl 0.0268 ev 0.53 | 6522 sps (1.5s+3.5s)
   eval: 512 games vs balanced/ows/road_builder: win 14.6% [11.8-17.9], avg VP 6.77
```

Leave the window open. To stop early, close the window: a checkpoint is saved every 25 updates
(`runs/main/checkpoints/`), and each one is a complete, usable agent.

## 3. How long it takes

About 4–5 s per update during the curriculum stages and longer once the self-play league starts (around
update 1,800), when frozen copies of the agent also have to be run. The whole run took about 16 hours on a
GTX 1660 Ti.

## 4. What to watch

- The **evaluation win rate against balanced/ows/road_builder** is the headline number. 25% is break-even
  at a four-player table; the agent beats the strategies if it's above 25% and the bracketed range doesn't
  include 25%. Each evaluation is only 512 games, so expect it to move by a few points between lines.
- Live charts: in a second terminal run `.\.venv\Scripts\tensorboard.exe --logdir runs\main\tb` and open
  http://localhost:6006.

## 5. If something goes wrong

| Problem | Fix |
|---|---|
| Window closed or PC restarted | Run the resume command again. At most ~25 updates are lost. |
| It crashed 5 times in a row and stopped | Read the last error in the window, close other apps and resume again. |
| `CUDA out of memory` | Close other GPU apps (games, video editing) and resume. |
| An error loading `latest.pt` | Restore the newest numbered checkpoint with the command below, then resume. |

```powershell
$newest = Get-ChildItem runs\main\checkpoints\update_*.pt | Sort-Object Name | Select-Object -Last 1
Copy-Item $newest.FullName runs\main\checkpoints\latest.pt -Force
```

## 6. Produce the results

The last checkpoint isn't always the strongest. Compare the evaluation lines (or the TensorBoard chart),
and if an earlier checkpoint was clearly better, pass `runs/main/checkpoints/update_XXXXX.pt` instead of
`latest.pt` below.

```powershell
# ~76,000 evaluation games, play-style analysis, replays, and the website data (web/data/summary.json)
.\.venv\Scripts\python.exe -m catanbot.pipeline agent --checkpoint runs/main/checkpoints/latest.pt --games 40000

# Re-run the training and agent notebooks
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks\03_training.ipynb notebooks\04_what_the_ai_learned.ipynb --ExecutePreprocessor.timeout=900

# Preview the website at http://localhost:3000
cd web
npm run dev
```

Every checkpoint is kept (~20 MB each, ~7 GB for the full run). Once you've picked the final agent, you can
delete the numbered ones:

```powershell
Get-ChildItem runs\main\checkpoints\update_*.pt | Remove-Item
```
