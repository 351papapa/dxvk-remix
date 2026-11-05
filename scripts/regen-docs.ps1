# Regenerates the generated reference documents from the source:
#   RtxOptions.md       from the RTX_OPTION declarations, by scripts/gen_rtx_options_md.py.
#   RemixApiSurface.md  from remix_c.h, by scripts-common/generate_remix_api_md.py.
[CmdletBinding()]
param(
  [switch]$SkipRtxOptions,
  [switch]$SkipApiSurface
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Generator {
  param(
    [string]$Script,
    [string]$Output
  )

  Write-Host "regen-docs: running $Script" -ForegroundColor Cyan
  & python $Script
  if ($LASTEXITCODE -ne 0) {
    Write-Error "$Script exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
  }

  if (-not (Test-Path $Output)) {
    Write-Error "$Output was not produced"
    exit 1
  }

  $size = (Get-Item $Output).Length
  Write-Host "regen-docs: regenerated $Output ($size bytes)" -ForegroundColor Green
}

if (-not $SkipRtxOptions) {
  Invoke-Generator -Script (Join-Path $RepoRoot 'scripts\gen_rtx_options_md.py') -Output (Join-Path $RepoRoot 'RtxOptions.md')
}

if (-not $SkipApiSurface) {
  Invoke-Generator -Script (Join-Path $RepoRoot 'scripts-common\generate_remix_api_md.py') -Output (Join-Path $RepoRoot 'RemixApiSurface.md')
}
