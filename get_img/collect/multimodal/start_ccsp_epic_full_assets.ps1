param(
    [string]$SimulatorPath = "",
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

if ($SimulatorPath) {
    $simulator = $SimulatorPath
} else {
    $simulatorCandidates = @(
        foreach ($packageRoot in Get-ChildItem -LiteralPath "E:\OpenHUTB" -Directory) {
            $installRoot = Join-Path `
                $packageRoot.FullName `
                "WindowsNoEditor\WindowsNoEditor"
            $egg = Join-Path `
                $installRoot `
                "PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
            $candidate = Join-Path $installRoot "CarlaUE4.exe"
            if (
                (Test-Path -LiteralPath $egg -PathType Leaf) -and
                (Test-Path -LiteralPath $candidate -PathType Leaf)
            ) {
                Get-Item -LiteralPath $candidate
            }
        }
    )
    $simulatorCandidates = @(
        $simulatorCandidates |
            Sort-Object -Property FullName -Unique
    )
    if ($simulatorCandidates.Count -eq 0) {
        throw "No CARLA 0.9.15 simulator matching the Python 3.7 API was found under E:\OpenHUTB."
    }
    if ($simulatorCandidates.Count -gt 1) {
        $candidateList = $simulatorCandidates.FullName -join [Environment]::NewLine
        throw "Multiple matching simulators were found. Pass -SimulatorPath explicitly:`n$candidateList"
    }
    $simulator = $simulatorCandidates[0].FullName
}

if (-not (Test-Path -LiteralPath $simulator -PathType Leaf)) {
    throw "Simulator executable not found: $simulator"
}
$workingDirectory = Split-Path -Parent $simulator

$simulatorProcessNames = @(
    "CarlaUE4",
    "CarlaUE4-Win64-Shipping"
)
$running = @(
    Get-Process -Name $simulatorProcessNames -ErrorAction SilentlyContinue
)
if ($running) {
    if (-not $Restart) {
        $processList = (
            $running |
                Sort-Object -Property Id |
                ForEach-Object { "$($_.ProcessName) pid=$($_.Id)" }
        ) -join ", "
        throw "CARLA is already running ($processList). Close it or run this script with -Restart."
    }
    $running | Stop-Process -Force
    $running | Wait-Process -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    $remaining = @(
        Get-Process -Name $simulatorProcessNames -ErrorAction SilentlyContinue
    )
    if ($remaining) {
        throw "CARLA processes are still running after restart cleanup."
    }
    Write-Host "Stopped all existing CARLA launcher and Shipping processes."
}

Start-Process `
    -FilePath $simulator `
    -WorkingDirectory $workingDirectory `
    -ArgumentList @(
        "-quality-level=Epic",
        "-NoTextureStreaming",
        "-USEALLAVAILABLECORES",
        '-ExecCmds="r.ForceLOD 0,foliage.ForceLOD 0,r.Streaming.FullyLoadUsedTextures 1"',
        "-carla-rpc-port=2000"
    )

Write-Host "Started CCSP with Epic quality and texture streaming disabled."
Write-Host "Executable: $simulator"
