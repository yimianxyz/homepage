# TEAM — homepage worker fleet

One lead + two side workers, all Claude Code (fable-5, ultracode), all
mounting THIS same /workspace. The lead manages; the sides execute.

| Member | Container | tmux reachable at | Home |
|---|---|---|---|
| lead (you, if reading from ai-homepage) | ai-homepage | (your own session) | own |
| side-a | ai-homepage-side-a | `tmux -S /workspace/.team/a.sock` (session `claude`) | own |
| side-b | ai-homepage-side-b | `tmux -S /workspace/.team/b.sock` (session `claude`) | own |

The sockets work across containers because all three share this
workspace bind-mount and run as the same uid. No docker access exists or
is needed. `/workspace/.team/` is gitignored scratch — sockets, handoff
notes, worktrees.

## Delegation doctrine (the point of the team)

Side workers have the SAME intelligence as the lead (same model, same
ultracode mode) and their tokens are CHEAPER. Therefore:

- **Default: delegate.** Any token-heavy task goes to a side worker
  first — bulk research/reading, large refactors, test-suite sweeps,
  codegen, doc passes, long debugging loops, multi-file migrations.
- **Lead keeps:** prioritisation, task decomposition, review, merges,
  small surgical edits, anything requiring the lead's conversation
  context, and talking to the operator.
- **Fallback:** do heavy work in the lead session only TEMPORARILY, when
  ALL side workers are at their usage limits — and hand it back when
  they recover. A side at its limit says so in its pane.
- Parallelise when tasks are independent: a on one, b on the other.

## Driving a side worker's tmux (cheat sheet)

```bash
S=/workspace/.team/a.sock                       # or b.sock
# is it busy? (busy = "esc to interrupt" visible; messages queue fine anyway)
tmux -S $S capture-pane -t claude -p | tail -15
# send a task (plain text: stage with -l, then ONE Enter as a separate call)
tmux -S $S send-keys -t claude -l 'Fix the flaky nav test; branch + PR per TEAM.md.'
tmux -S $S send-keys -t claude Enter
# slash commands need TWO Enters (first is eaten by autocomplete)
tmux -S $S send-keys -t claude -l '/status'
tmux -S $S send-keys -t claude Enter; tmux -S $S send-keys -t claude Enter
# read more scrollback
tmux -S $S capture-pane -t claude -p -S -200
```

Gotchas (hard-won): long text collapses to `[Pasted text #N]` in the box —
normal, still submits. `❯ old text` above an empty box is a transcript
echo, not a draft. `/status` is the authoritative model check; prose
self-reports are unreliable. After tasking, capture-pane to confirm the
message was accepted (working or queued) before assuming it landed.

## Channels — pick by weight

1. **tmux** — tasking, nudges, quick status. Ephemeral.
2. **`/workspace/.team/`** — artifacts, scratch handoffs, anything that
   doesn't belong in git history. Gitignored.
3. **GitHub (the org channel)** — every delegated task of substance gets
   an issue (label `side-a`/`side-b`); the side delivers a PR
   referencing it; the lead reviews and merges. This is the durable
   record of who did what and why.

## Concurrency — one writer per working tree

Three agents share one checkout; uncoordinated writes WILL corrupt work.

- The main checkout (`/workspace`) belongs to the LEAD. Sides never
  commit, switch branches, or mutate the index there.
- Sides do mutating work in their own worktree:
  `git worktree add /workspace/.team/wt-<task> -b side-a/<task>` then
  work, commit, push from inside it; remove the worktree when merged.
- Branch namespace: `side-a/*`, `side-b/*`. Normal pushes only (the
  ruleset blocks force-push anyway).
- Read-only work (research, review, running tests on a fresh worktree)
  needs no coordination.

## Health & limits

- A side that errors or hits usage limits: note it, reassign or wait.
- Sides run with `--restart unless-stopped` containers + `--init`; if a
  side's session dies, the operator can relaunch it (`--continue`
  preserves its context). Report persistent breakage to the operator.
