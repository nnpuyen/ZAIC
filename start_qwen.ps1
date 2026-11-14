param(
    [string]$Model = "",
    [string]$Server = "llama-server",
    [int]$Port = 8000,
    [int]$ContextSize = 32768,
    [string]$Alias = ""
)

if ([string]::IsNullOrWhiteSpace($Model)) {
    $models = @(Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot "models\Qwen3.5-9B") -Filter "*.gguf" -File)
    if ($models.Count -eq 0) {
        throw "No GGUF model found in $PSScriptRoot\models\Qwen3.5-9B. Pass -Model with the model path."
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
    $Alias = [System.IO.Path]::GetFileNameWithoutExtension($Model)
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
