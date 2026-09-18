<#
.SYNOPSIS
    Packages GPBSF code and dataset into POSIX-compliant ZIP archives ready for Kaggle Dataset upload.

.DESCRIPTION
    Creates clean ZIP packages with POSIX forward slashes ('/'):
    1. gpbsf_code.zip: Code, configurations, submodels (excluding git history, cache, runs, and local data).
    2. gpbsf_dataset_v4.zip: The BGSPCD-v4-robust dataset folder.

.PARAMETER DatasetPath
    Absolute or relative path to the bgspcd_v4_robust folder.

.PARAMETER OutputDir
    Destination folder where ZIP files and Kaggle metadata will be generated. Default: ./kaggle_assets
#>

param(
    [string]$DatasetPath = "",
    [string]$OutputDir = "kaggle_assets"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$AssetsDir = Join-Path $ProjectRoot $OutputDir

if (-not (Test-Path $AssetsDir)) {
    New-Item -ItemType Directory -Path $AssetsDir -Force | Out-Null
}

function Create-PosixZip {
    param(
        [string]$SourcePath,
        [string]$DestinationZip,
        [string[]]$Excludes,
        [System.IO.Compression.CompressionLevel]$Level = [System.IO.Compression.CompressionLevel]::Optimal
    )

    if (Test-Path $DestinationZip) {
        Remove-Item -Force $DestinationZip
    }

    $archive = [System.IO.Compression.ZipFile]::Open($DestinationZip, [System.IO.Compression.ZipArchiveMode]::Create)
    $sourceNormalized = $SourcePath.TrimEnd("\", "/")
    $count = 0

    try {
        Get-ChildItem -Path $sourceNormalized -Recurse -File | ForEach-Object {
            $file = $_
            $rel = $file.FullName.Substring($sourceNormalized.Length).TrimStart("\", "/")

            $skip = $false
            if ($null -ne $Excludes) {
                foreach ($pattern in $Excludes) {
                    if ($rel -match $pattern) {
                        $skip = $true
                        break
                    }
                }
            }

            if (-not $skip) {
                # CRITICAL: Convert backslashes to forward slashes for Linux/Kaggle compatibility
                $entryName = $rel.Replace("\", "/")
                [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $file.FullName, $entryName, $Level) | Out-Null
                $count++
            }
        }
    }
    finally {
        $archive.Dispose()
    }

    return $count
}

Write-Host "=== 1. Packaging GPBSF Code (POSIX Normalized) ===" -ForegroundColor Cyan
$CodeZipPath = Join-Path $AssetsDir "gpbsf_code.zip"

$ExcludePatterns = @(
    "(^|[\\/])\.git($|[\\/])",
    "__pycache__",
    "\.pyc$",
    "^data($|[\\/])",
    "^runs($|[\\/])",
    "kaggle_assets",
    "\.zip$"
)

$fileCount = Create-PosixZip -SourcePath $ProjectRoot -DestinationZip $CodeZipPath -Excludes $ExcludePatterns -Level Optimal
Write-Host "Packaged $fileCount files into $CodeZipPath" -ForegroundColor Green

# Kaggle Metadata for Code
$CodeMetadata = @{
    title = "gpbsf-code"
    id = "[YOUR_KAGGLE_USERNAME]/gpbsf-code"
    licenses = @(@{ name = "CC0-1.0" })
}
$CodeMetadata | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $AssetsDir "dataset-metadata-code.json") -Encoding utf8

# 2. Package Dataset if provided
if ($DatasetPath -ne "" -and (Test-Path $DatasetPath)) {
    Write-Host "`n=== 2. Packaging BGSPCD Dataset (POSIX Normalized) ===" -ForegroundColor Cyan
    $DatasetResolved = (Resolve-Path $DatasetPath).Path
    $DatasetZipPath = Join-Path $AssetsDir "gpbsf_dataset_v4.zip"

    $dataCount = Create-PosixZip -SourcePath $DatasetResolved -DestinationZip $DatasetZipPath -Excludes $null -Level Fastest
    Write-Host "Packaged $dataCount files into $DatasetZipPath" -ForegroundColor Green

    $DataMetadata = @{
        title = "gpbsf-dataset-v4"
        id = "[YOUR_KAGGLE_USERNAME]/gpbsf-dataset-v4"
        licenses = @(@{ name = "CC0-1.0" })
    }
    $DataMetadata | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $AssetsDir "dataset-metadata-dataset.json") -Encoding utf8
} else {
    Write-Host "`n[Notice] DatasetPath not provided or not found. If packaging dataset, run:" -ForegroundColor Yellow
    Write-Host ".\prepare_kaggle_assets.ps1 -DatasetPath <Path-to-bgspcd_v4_robust>" -ForegroundColor Yellow
}

Write-Host "`n=== Packaging Complete ===" -ForegroundColor Cyan
Write-Host "Output Directory: $AssetsDir"
