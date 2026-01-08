# Fork Maintenance Guide

This document explains how we maintain the `kaiko-ai/verl` fork of `volcengine/verl`.

## Branching Strategy

We use a dual-branch strategy to separate upstream syncs from custom changes:

- **`main`** - Pinned upstream version + our custom changes on top
- **`upstream-track`** - Pure upstream mirror (no custom code), marks which upstream version we're based on

## Workflow Types

### 1. Syncing Upstream Changes

When updating to a newer upstream version:

1. Update `upstream-track` to point to the desired upstream commit:
   ```bash
   git checkout upstream-track
   git fetch upstream
   git reset --hard upstream/main  # or specific commit
   git push origin upstream-track --force
   ```

2. Create a PR from `upstream-track` to `main`
3. Merge using **merge commit** (preserves upstream history)

This keeps upstream updates separate and easy to identify.

### 2. Adding Custom Changes

When adding fork-specific modifications:

1. Create a feature branch from `main`
2. Make your custom changes
3. Create a PR targeting `main`
4. Merge using **squash merge** (keeps custom changes as single commits)

This keeps custom changes small, reviewable, and easy to track.

## Checking Custom Changes

To see all custom changes we've made on top of upstream:

```bash
git log upstream-track..main --oneline
```

This shows only the commits that differ from the upstream version we're based on.

## Current State

- **Upstream base**: The version in `upstream-track` shows which upstream commit we're currently based on
- **Custom commits**: Use the command above to see our modifications
