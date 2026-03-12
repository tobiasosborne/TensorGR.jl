# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.
See CLAUDE.md for full architecture and API reference.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Session Handoff (2026-03-09)

### What was done this session
- **solve_tensors**: Linear tensor equation solver (TGR-9lf closed)
- **Documentation overhaul**: CLAUDE.md + 10 API docs + 7 tutorial sections (2,200+ lines)
- **Symbolic components**: SymbolicMetric + Christoffel/Riemann/Ricci pipeline via Symbolics.jl (TGR-j6f closed)
- **Wald verification**: 7 example scripts verifying GR identities (TGR-6v7 closed)
- **Schwarzschild + FLRW**: Symbolic verification (31 checks total, TGR-8pk + TGR-3gm closed)
- **Pattern indices**: `down(:a_)` pattern variables in RewriteRule (TGR-xd0 closed)
- **Smooth maps**: define_mapping!, pullback, pushforward (TGR-6rp closed)
- **Course verification**: 8 scripts covering lectures 7-22 (73+ checks)

### Current state
- **3,534 tests pass**, 152 benchmarks pass
- 7 issues closed this session, 9 open (run `bd ready`)

### Open issues (run `bd ready` for current list)
- TGR-byb (P2): BinaryBuilder for xperm.c
- TGR-erv (P2): Pkg registration
- TGR-v4v (P3): Product manifolds
- TGR-1kw (P3): Submanifolds/boundaries
- TGR-38d (P4): Invar database
- TGR-dhp (P3): TOV equation solver
- TGR-61p (P3): Geodesic ODE integration
- TGR-293h (P4): Symmetry-reduced metric ansatz
- TGR-6rp: in_progress but closed — may show stale

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds


<!-- BEGIN BEADS INTEGRATION -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Dolt-powered version control with native sync
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update <id> --claim --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task atomically**: `bd update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs via Dolt:

- Each write auto-commits to Dolt history
- Use `bd dolt push`/`bd dolt pull` for remote sync
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

<!-- END BEADS INTEGRATION -->
