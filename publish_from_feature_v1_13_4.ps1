$ErrorActionPreference = "Stop"
$ExpectedBranch = "feature/1.13.4-dashboard-summary"
$Version = "1.13.4"

$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne $ExpectedBranch) {
    throw "Expected branch '$ExpectedBranch', but current branch is '$currentBranch'."
}

python -m compileall -q src tests
if ($LASTEXITCODE -ne 0) { throw "Python compilation failed." }

python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }

git status
git add -A
git commit -m "feat(api): add sanitized dashboard read endpoints"
git push -u origin $ExpectedBranch
Write-Host "API v$Version published from $ExpectedBranch"
