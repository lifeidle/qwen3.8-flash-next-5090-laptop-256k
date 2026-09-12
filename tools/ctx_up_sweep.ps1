# ctx sweep (no MTP): step up context from 112K, see how far it goes
# Measures with temperature=0 + fixed 512 tokens (server log eval time is authoritative)
$model  = "D:\models\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64\Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-00001-of-00028.gguf"
$mmproj = "D:\models\Qwen3.8-Flash-Next\mmproj-Qwen3.8-Flash-Next-F16.gguf"
$tpl    = "D:\llama.cpp\templates\qwen38-fixed.jinja"
$outFile = "C:\Users\chenhua\Desktop\1\ctx_up_result.txt"
$logFile = "C:\Users\chenhua\Desktop\1\ctx_up.log"

$configs = @(
    @{ ctx = 131072;  ncmoe = 32; tag = "c128K_moe32" },
    @{ ctx = 147456;  ncmoe = 32; tag = "c144K_moe32" },
    @{ ctx = 163840;  ncmoe = 33; tag = "c160K_moe33" },
    @{ ctx = 196608;  ncmoe = 34; tag = "c192K_moe34" },
    @{ ctx = 262144;  ncmoe = 36; tag = "c256K_moe36" }
)

$results = @()
$results += "ctx sweep WITHOUT MTP (baseline: 112K/moe32 = 27.12 tok/s, VRAM 23.2GB)"
$results += ""

foreach ($cfg in $configs) {
    Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 4
    Remove-Item $logFile -Force -ErrorAction SilentlyContinue

    $argList = @(
        "-m", $model, "-ngl", "99", "--n-cpu-moe", "$($cfg.ncmoe)",
        "-fa", "on", "-fit", "off", "-c", "$($cfg.ctx)", "-np", "1",
        "--jinja", "--alias", "qwen3.8-flash-next",
        "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
        "--host", "127.0.0.1", "--port", "8080",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--chat-template-file", $tpl,
        "--load-mode", "dio",
        "--log-file", $logFile,
        "-mm", $mmproj
    )

    try {
        $proc = Start-Process -FilePath "D:\llama.cpp\llama-server.exe" -ArgumentList $argList -WindowStyle Hidden -PassThru
    } catch {
        $results += "$($cfg.tag) : Start-Process failed"
        continue
    }

    $ready = $false
    for ($i = 0; $i -lt 45; $i++) {
        try { $h = Invoke-WebRequest "http://127.0.0.1:8080/health" -TimeoutSec 3 -UseBasicParsing; if ($h.StatusCode -eq 200) { $ready = $true; break } } catch {}
        Start-Sleep -Seconds 3
    }
    if (-not $ready) {
        $results += "$($cfg.tag) | ctx=$($cfg.ctx) moe=$($cfg.ncmoe) | STARTUP FAILED (OOM)"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        continue
    }

    $vram = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>&1 | Out-String).Trim()

    $body = '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Please explain the history of artificial intelligence, including main schools, key breakthroughs and representative achievements."}],"max_tokens":512,"temperature":0,"reasoning_effort":"low"}'
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:8080/v1/chat/completions" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 1800
        $ct = $r.usage.completion_tokens
    } catch { $ct = "req failed" }

    Start-Sleep -Seconds 2
    $log = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
    $ms = [regex]::Matches($log, 'eval time =\s*([\d.]+) ms /\s*(\d+) tokens \([^)]*?,\s*([\d.]+) tokens per second')
    if ($ms.Count -gt 0 -and $ms[$ms.Count-1].Groups.Count -ge 4) {
        $last = $ms[$ms.Count-1]
        $results += "$($cfg.tag) | ctx=$($cfg.ctx) moe=$($cfg.ncmoe) | decode=$($last.Groups[3].Value) t/s | out=$ct | VRAM=$vram"
    } else {
        $results += "$($cfg.tag) | ctx=$($cfg.ctx) moe=$($cfg.ncmoe) | no eval-time | out=$ct | VRAM=$vram"
    }

    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
$results += ""
$results += "done"
$results | Set-Content $outFile -Encoding UTF8
Write-Output "CTX UP SWEEP DONE"
