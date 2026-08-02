#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BRANCH="feature/1.13.0-multihorizon-series-movements"
TAG="api-v1.13.0"

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "Current branch is '$CURRENT_BRANCH'. Switch to '$EXPECTED_BRANCH' before publishing."
  exit 1
fi

git add .
git commit -m "feat: add multi-horizon series movement model v1.13.0"
git push -u origin "$EXPECTED_BRANCH"
git tag -a "$TAG" -m "Market Cycle Trader API v1.13.0"
git push origin "$TAG"

echo "Published branch: $EXPECTED_BRANCH"
echo "Published tag: $TAG"
