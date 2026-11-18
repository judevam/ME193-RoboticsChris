# My workflow: WSL + Tufts HPC (cheat sheet)

Written for me, for coming back after a break. Full details: [README.md](README.md).

## Every session

1. **Connect the Tufts VPN** (if off campus).
2. **Open WSL** (the Ubuntu app, or VSCode > "Connect to WSL"). Not PowerShell.
3. **Go to the folder** and check you are in the right place:
   ```
   cd ~/ME193-Robotics/"Public stuff"/mujoco-hpc     # wherever my WSL clone lives
   pwd
   git branch                                        # should say JudeBranch
   ```
4. **Log in / re-check everything** (password + Duo; the login lasts 2 hours):
   ```
   python3 setup.py
   ```
   All PASS = the cluster is ready. Use `--skip-gpu` to be quicker.

## Run something

```
python3 runIt.py --gpu scripts/09_paddleDemo.py     # train the paddle, makes paddle.mp4 (GPU node renders the video)
python3 runIt.py --gpu scripts/08_demo.py           # the bouncing demo
python3 runIt.py --gpu --time 01:00:00 myscript.py  # longer job
```
Results (videos, `.pt`, `.csv`) come back to `runs/<name>/` and videos open on their own.
`runIt.py` copies the script fresh every time, so edits always take effect.

| Need | Command |
|---|---|
| see my jobs | `python3 status.py` (add `--watch` to follow) |
| stop a job | `python3 status.py --cancel <jobid>` |
| fetch results after Ctrl-C | `python3 status.py --fetch <jobid>` |
| open a shell on the cluster | `ssh hpc` (`exit` to leave) |
| disk full | `python3 cleanup.py` |
| everything broken | `python3 setup.py clean --dry-run`, then `python3 setup.py clean` |

## Getting my edits into WSL

My edits are made on Windows (VSCode, OneDrive folder). WSL has its own clone, so it does not see them until I pull.
`origin` is Chris's repo and I cannot push to it (403), so go through my fork:

```
# one time, on Windows: fork the repo on GitHub as judevam, then
git remote add fork https://github.com/judevam/ME193-Robotics.git

# every time, on Windows
git add <files>; git commit -m "..."
git push fork JudeBranch

# then in WSL
git pull fork JudeBranch
```
To get Chris's newest changes: `git fetch origin` then `git merge origin/main` (while on JudeBranch).

Quick way with no git: run from the Windows copy in WSL,
`cd "/mnt/c/Users/jetv2/OneDrive/Documents/GitHub/ME193-RoboticsChris/Public stuff/mujoco-hpc"`.
It works but is slower and can hit permission/line-ending quirks, so the WSL clone is better.

## When something is off

- **Running an old version of a script:** I am in the wrong copy. Check with `grep -n <something I changed> scripts/<file>.py`.
- **Can't reach the cluster / login fails:** VPN off, or wrong `HPC_USER` / `HPC_HOST` in `config.sh`.
- **`no such file`:** not in the `mujoco-hpc` folder, or forgot the `scripts/` prefix.
- **Job stays PENDING:** waiting for a node (normal for GPU). `python3 status.py` shows why.
- **Still stuck:** send the newest file in `logs/`.
