#!/usr/bin/env bash
set -euo pipefail

# Safely remove obsolete versioned research wrappers after their valid logic has
# been consolidated into stable filenames. By default this script only checks.
# Use --apply to execute git rm for files that have no remaining references.

MODE="check"
if [[ "${1:-}" == "--apply" ]]; then
  MODE="apply"
elif [[ -n "${1:-}" && "${1:-}" != "--check" ]]; then
  echo "Usage: $0 [--check|--apply]" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "ERROR: run this script inside the market_cycle_trader_api Git repository." >&2
  exit 2
fi
cd "$REPO_ROOT"

TARGETS=(
  "scripts/research_asset_rotation_leadership_v12.py"
  "scripts/research_asset_rotation_leadership_v13.py"
  "scripts/research_asset_rotation_independent_validation_v101.py"
  "scripts/research_asset_rotation_independent_validation_v103.py"
)

if [[ "$MODE" == "apply" ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked working tree changes detected. Commit/stash them before cleanup." >&2
    exit 2
  fi
fi

blocked=0
removable=0
missing=0

echo "Research wrapper cleanup"
echo "Mode: $MODE"
echo

for target in "${TARGETS[@]}"; do
  if [[ ! -e "$target" ]]; then
    echo "[MISSING] $target"
    missing=$((missing + 1))
    continue
  fi

  base="$(basename "$target")"
  refs="$(git grep -n -F "$base" -- . \
    ":!$target" \
    ":!scripts/cleanup_research_wrappers.sh" 2>/dev/null || true)"

  if [[ -n "$refs" ]]; then
    echo "[BLOCKED] $target"
    echo "  Remaining references:"
    while IFS= read -r line; do
      [[ -n "$line" ]] && echo "    $line"
    done <<< "$refs"
    blocked=$((blocked + 1))
    echo
    continue
  fi

  echo "[READY]   $target"
  removable=$((removable + 1))
  if [[ "$MODE" == "apply" ]]; then
    git rm -- "$target"
  fi
  echo
 done

echo "Summary: ready=$removable blocked=$blocked missing=$missing"

if [[ "$MODE" == "check" ]]; then
  echo
  if (( blocked > 0 )); then
    echo "Nothing blocked will be deleted. Consolidate/update the references above first."
  fi
  echo "When every desired file shows [READY], run:"
  echo "  bash scripts/cleanup_research_wrappers.sh --apply"
else
  echo
  if (( blocked > 0 )); then
    echo "Cleanup partially applied. Blocked files were preserved."
  else
    echo "Cleanup staged with git rm. Review with:"
    echo "  git status --short"
    echo "  git diff --cached --stat"
  fi
fi

# A non-zero exit on blocked files makes this suitable for CI/pre-cleanup checks.
if (( blocked > 0 )); then
  exit 1
fi
