param(
    [string]$Model = "",
    [string]$Server = "llama-server",
    [int]$Port = 8000,
    [int]$ContextSize = 32768,
    [string]$Alias = ""
)

if ([string]::IsNullOrWhiteSpace($Model)) {
    $preferredDirectory = Join-Path $PSScriptRoot "models\Gemma"
    $models = @(Get-ChildItem -LiteralPath $preferredDirectory -Filter "*.gguf" -File -ErrorAction SilentlyContinue)
    if ($models.Count -eq 0) {
        throw "No GGUF model found in $preferredDirectory. Put the Gemma GGUF there or pass -Model with its path."
    }
    if ($models.Count -gt 1) {
        throw "Multiple GGUF models found. Pass -Model explicitly: $($models.FullName -join ', ')"
    }
    $Model = $models[0].FullName
}

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Model file not found: $Model"
}

if ([string]::IsNullOrWhiteSpace($Alias)) {
    if ((Split-Path -Leaf (Split-Path -Parent $Model)) -ieq "Gemma") {
        $Alias = "gemma-local"
    } else {
        $Alias = [System.IO.Path]::GetFileNameWithoutExtension($Model)
    }
}

& $Server `
    "--model" $Model `
    "--host" "127.0.0.1" `
    "--port" $Port `
    "--ctx-size" $ContextSize `
    "--parallel" "1" `
    "--reasoning" "off" `
    "--alias" $Alias `
    "--jinja"
