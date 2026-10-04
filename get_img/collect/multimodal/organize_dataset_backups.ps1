param(
    [string]$Root = "E:\pythonProject\air_groud\get_img",
    [string]$DatasetName = "dataset_uav_multimap_town600_hutb300_ccsp300"
)

$ErrorActionPreference = "Stop"
$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
$backupBase = [IO.Path]::GetFullPath((Join-Path $resolvedRoot "dataset_backups"))
if (-not $backupBase.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Backup destination is outside the requested root: $backupBase"
}

New-Item -ItemType Directory -Path $backupBase -Force | Out-Null
$prefix = "${DatasetName}_sequence_backup_"
$oldRoots = Get-ChildItem -LiteralPath $resolvedRoot -Directory |
    Where-Object { $_.Name.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) } |
    Sort-Object Name

$moved = @()
foreach ($oldRoot in $oldRoots) {
    $oldFull = [IO.Path]::GetFullPath($oldRoot.FullName)
    if (-not $oldFull.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Backup source is outside the requested root: $oldFull"
    }

    $stamp = $oldRoot.Name.Substring($prefix.Length)
    foreach ($mapDir in @(Get-ChildItem -LiteralPath $oldFull -Directory -Force)) {
        $mapParent = [IO.Path]::GetFullPath((Join-Path $backupBase $mapDir.Name))
        $target = [IO.Path]::GetFullPath((Join-Path $mapParent $stamp))
        if (-not $target.StartsWith($backupBase, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Backup target is outside the backup directory: $target"
        }
        if (Test-Path -LiteralPath $target) {
            throw "Backup target already exists: $target"
        }

        New-Item -ItemType Directory -Path $mapParent -Force | Out-Null
        Move-Item -LiteralPath $mapDir.FullName -Destination $target
        $moved += [pscustomobject]@{
            Map = $mapDir.Name
            Timestamp = $stamp
            Destination = $target
        }
    }

    $remaining = @(Get-ChildItem -LiteralPath $oldFull -Force)
    if ($remaining.Count -gt 0) {
        $metadataParent = Join-Path $backupBase "_run_metadata"
        $metadataTarget = [IO.Path]::GetFullPath((Join-Path $metadataParent $stamp))
        if (-not $metadataTarget.StartsWith($backupBase, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Metadata target is outside the backup directory: $metadataTarget"
        }
        New-Item -ItemType Directory -Path $metadataTarget -Force | Out-Null
        foreach ($item in $remaining) {
            Move-Item -LiteralPath $item.FullName -Destination $metadataTarget
        }
    }

    if (@(Get-ChildItem -LiteralPath $oldFull -Force).Count -ne 0) {
        throw "Old backup directory is not empty: $oldFull"
    }
    Remove-Item -LiteralPath $oldFull
}

$moved | Sort-Object Map, Timestamp | Format-Table -AutoSize
Write-Output "Moved entries: $($moved.Count)"
Write-Output "Backup base: $backupBase"
