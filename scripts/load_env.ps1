<#!
.SYNOPSIS
    Load key=value pairs from a .env file into the current PowerShell session.

.DESCRIPTION
    Parses a .env (default: project root .env) ignoring blank lines and comments (# ...).
    Each KEY=VALUE line is exported into the session via $env:KEY = VALUE.

.PARAMETER Path
    Optional path to the .env file (defaults to ../.env relative to this script).

.EXAMPLE
    # Dot-source to persist variables in your current session
    . ./scripts/load_env.ps1

.EXAMPLE
    # Custom env file path
    . ./scripts/load_env.ps1 -Path ./.env.local

.NOTES
    - Secrets are NOT echoed.
    - Use dot-sourcing (leading dot + space) so variables survive after the script ends.
    - This script never commits the .env file; ensure .env stays in .gitignore.
#>

param(
    [string]$Path = (Join-Path (Split-Path $PSScriptRoot -Parent) '.env')
)

if (-not (Test-Path -LiteralPath $Path)) {
    Write-Warning "Env file not found: $Path"; return
}

$loaded = 0
Get-Content -LiteralPath $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line) { return }
    if ($line.StartsWith('#')) { return }
    # Allow inline comments: KEY=VALUE # comment
    $parts = $line -split '#',2
    $kv = $parts[0]
    $eqIndex = $kv.IndexOf('=')
    if ($eqIndex -lt 1) { return }
    $key = $kv.Substring(0,$eqIndex).Trim()
    $val = $kv.Substring($eqIndex+1).Trim()
    if (-not $key) { return }
    $env:$key = $val
    $loaded++
}

Write-Host "[load_env] Loaded $loaded variable(s) from $Path" -ForegroundColor Green
Write-Host "[load_env] (Values hidden for safety)" -ForegroundColor DarkGray
