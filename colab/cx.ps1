param([string]$File, [int]$Timeout = 300)
$w = "C:/Users/ptkac/.claude/plugins/cache/colab-cli-ops/colab-cli-ops/0.1.0/skills/colab-cli-ops/scripts/colab.ps1"
powershell -File $w exec -s tswarm --timeout $Timeout -f $File 2>&1 | Out-String
