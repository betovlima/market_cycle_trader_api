$ErrorActionPreference = "Stop"

$ExpectedBranch = "feature/1.13.3-winner-period"
$CommitMessage = "feat(api): lock the backtest period to winner v1.13.2"

$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne $ExpectedBranch) {
    throw "Current branch is '$currentBranch'. Switch to '$ExpectedBranch' before publishing."
}

python -m compileall -q src tests
if ($LASTEXITCODE -ne 0) {
    throw "Python compilation failed."
}

python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed."
}

git add -A -- .
$staged = git diff --cached --name-only
if (-not $staged) {
    throw "No API changes are staged. Copy the release files into the repository first."
}

git commit -m $CommitMessage
git push -u origin $ExpectedBranch

Write-Host "Published branch $ExpectedBranch successfully."
