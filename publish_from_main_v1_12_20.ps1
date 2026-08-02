$ErrorActionPreference = "Stop"

$expectedVersion = "1.12.20"
$tag = "v$expectedVersion"

if (-not (Test-Path "./pyproject.toml")) {
    throw "Run this script from the market_cycle_trader_api repository root."
}

$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne "main") {
    git checkout main
}

git pull origin main

python -m compileall -q src scripts
if ($LASTEXITCODE -ne 0) {
    throw "Python validation failed."
}

$version = (Get-Content ./VERSION -Raw).Trim()
if ($version -ne $expectedVersion) {
    throw "VERSION is '$version'; expected '$expectedVersion'."
}

$existingTag = git tag --list $tag
if ($existingTag) {
    throw "Tag $tag already exists. Do not overwrite an existing release tag."
}

git add `
    .python-version `
    VERSION `
    pyproject.toml `
    railway.toml `
    requirements.txt `
    run_local.ps1 `
    run_local.sh `
    RELEASE_V1_12_20.md `
    VALIDATION_V1_12_20.txt `
    publish_from_main_v1_12_20.ps1 `
    scripts `
    src

$staged = git diff --cached --name-only
if ($staged -match '(^|/)(\.env|\.env\..+|\.env\.example|\.gitignore)$') {
    git reset
    throw "A prohibited environment or ignore file was staged. Nothing was committed."
}

if (-not $staged) {
    throw "No changes were staged."
}

git commit -m "release: API v1.12.20 restore Mongo URL runtime and admin endpoints"
git push origin main

git tag -a $tag -m "API v1.12.20 - Mongo URL runtime fix"
git push origin $tag

git status
git log --oneline --decorate -10
