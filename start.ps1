<#
.SYNOPSIS
  Prepara o ambiente e sobe a interface web do YARGen.

.DESCRIPTION
  Existe para evitar a armadilha mais comum no Windows: `py` so existe se o
  Python Launcher estiver instalado, e o Python da Microsoft Store nao o
  instala. Sem ele o comando falha, o ambiente virtual nao e criado, e o
  `pip install` seguinte acaba indo para o Python global - com os executaveis
  numa pasta que nao esta no PATH. O resultado e um "instalou tudo mas nada
  roda" que nao explica a propria causa.

  Este script procura um Python utilizavel, cria o ambiente virtual se ainda
  nao existir, instala o que falta e sobe o servidor.

.PARAMETER Full
  Instala tambem Demucs (separacao de stems) e faster-whisper (letra). Sem
  isto o chart sai charteando a mixagem inteira, e em trap nao funciona.
  Usa o indice CPU-only do PyTorch: ~1 GB em vez de ~6 GB com CUDA, que num
  PC sem placa NVIDIA nao serve para nada.

.PARAMETER Port
  Porta do servidor. Padrao 8000.

.EXAMPLE
  .\start.ps1
  .\start.ps1 -Full
  .\start.ps1 -Port 8080
#>
[CmdletBinding()]
param(
  [switch]$Full,
  [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
# No PowerShell 7.4+ isto vem como $true e faz um comando nativo que retorna
# codigo diferente de zero virar excecao. As sondas abaixo ("esse pacote ja
# esta instalado?") dependem de ler $LASTEXITCODE sem estourar.
$PSNativeCommandUseErrorActionPreference = $false
Set-Location -Path $PSScriptRoot

function Find-Python {
  foreach ($candidate in @('py -3', 'python', 'python3')) {
    $parts = $candidate.Split(' ')
    $exe = $parts[0]
    # NAO usar $args aqui: e variavel automatica do PowerShell e sobrescreve-la
    # dentro de uma funcao da comportamento imprevisivel.
    $pyArgs = if ($parts.Count -gt 1) { $parts[1..($parts.Count - 1)] } else { @() }
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    try {
      $version = & $exe @pyArgs -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
    } catch { continue }
    # O "python" do Windows sem Python instalado e um atalho que abre a Loja
    # e nao imprime nada: a checagem de versao vazia filtra esse caso.
    if (-not $version) { continue }
    $parsed = [version]$version
    if ($parsed -lt [version]'3.10') {
      Write-Host "  $exe e $version - o projeto precisa de 3.10 ou mais novo" -ForegroundColor DarkYellow
      continue
    }
    return @{ Exe = $exe; Args = $pyArgs; Version = $version }
  }
  return $null
}

Write-Host "YARGen" -ForegroundColor Cyan

$python = Find-Python
if (-not $python) {
  Write-Host @"

Nenhum Python 3.10+ utilizavel foi encontrado.

Instale de https://www.python.org/downloads/ e marque a caixa
"Add python.exe to PATH" durante a instalacao. Depois abra um PowerShell
NOVO (o PATH so vale em janelas abertas depois da instalacao) e rode este
script de novo.
"@ -ForegroundColor Red
  exit 1
}
Write-Host "  Python $($python.Version) encontrado" -ForegroundColor DarkGray

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
  Write-Host "  criando o ambiente virtual em .venv ..." -ForegroundColor DarkGray
  & $python.Exe @($python.Args) -m venv .venv
  if (-not (Test-Path $venvPython)) {
    Write-Host "Falhou ao criar o ambiente virtual em .venv" -ForegroundColor Red
    exit 1
  }
}

# Instala so quando falta algo: relancar o servidor nao deve reinstalar nada.
$needsInstall = $true
try {
  & $venvPython -c 'import yargen, fastapi' 2>$null | Out-Null
  $needsInstall = ($LASTEXITCODE -ne 0)
} catch { $needsInstall = $true }

if ($needsInstall) {
  Write-Host "  instalando dependencias (a primeira vez demora) ..." -ForegroundColor DarkGray
  & $venvPython -m pip install --upgrade pip --quiet
  & $venvPython -m pip install -e ".[web]"
  if ($LASTEXITCODE -ne 0) { Write-Host "Falhou a instalacao" -ForegroundColor Red; exit 1 }
}

if ($Full) {
  $needsFull = $true
  try {
    & $venvPython -c 'import demucs, faster_whisper' 2>$null | Out-Null
    $needsFull = ($LASTEXITCODE -ne 0)
  } catch { $needsFull = $true }
  if ($needsFull) {
    Write-Host "  instalando Demucs e faster-whisper (~1 GB, demora) ..." -ForegroundColor DarkGray
    # Indice CPU-only: a build padrao arrasta ~5 GB de CUDA que nao tem uso
    # nenhum sem uma placa NVIDIA.
    & $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    & $venvPython -m pip install -e ".[all,web]"
    if ($LASTEXITCODE -ne 0) { Write-Host "Falhou a instalacao" -ForegroundColor Red; exit 1 }
  }
}

Write-Host ""
Write-Host "  http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "  Ctrl+C para parar" -ForegroundColor DarkGray
Write-Host ""
& $venvPython -m webapp.server --port $Port
