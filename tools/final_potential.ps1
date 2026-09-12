# Final potential sweep @ 256K: test remaining untried flags
$model  = "D:\models\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64\Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-00001-of-00028.gguf"
$mmproj = "D:\models\Qwen3.8-Flash-Next\mmproj-Qwen3.8-Flash-Next-F16.gguf"
$tpl    = "D:\llama.cpp\templates\qwen38-fixed.jinja"
$outFile = "C:\Users\chenhua\Desktop\1\final_potential.txt"
$logFile = "C:\Users\chenhua\Desktop\1\final_pot.log"

$configs = @(
    @{ tag = "256K_base";        extra = @() },
    @{ tag = "256K_cpustrict";   extra = @("--cpu-strict", "1") },
    @{ tag = "256K_prio2";       extra = @("--prio", "2") },
    @{ tag = "256K_schedbatch";  extra = @("-b", "4096") }
)

$results = @()
$results += "Final potential sweep @ 256K (no MTP). Baseline: 25.08 tok/s"
$results += ""

foreach ($cfg in $configs) {
    Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 4
    Remove-Item $logFile -Force -ErrorAction SilentlyContinue

    $argList = @(
        "-m", $model, "-ngl", "99", "--n-cpu-moe", "36",
        "-fa", "on", "-fit", "off", "-c", "262144", "-np", "1",
        "--jinja", "--alias", "qwen3.8-flash-next",
        "--temp", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
        "--host", "127.0.0.1", "--port", "8080",
        "-ctk", "q8_0", "-ctv", "q8_0",
        "--chat-template-file", $tpl,
        "--load-mode", "dio",
        "--log-file", $logFile,
        "-mm", $mmproj
    )
    $argList += $cfg.extra

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
        $results += "$($cfg.tag) | STARTUP FAILED"
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
        $results += "$($cfg.tag) | decode=$($last.Groups[3].Value) t/s | out=$ct | VRAM=$vram | flags=$($cfg.extra -join ' ')"
    } else {
        $results += "$($cfg.tag) | no eval-time | out=$ct | VRAM=$vram | flags=$($cfg.extra -join ' ')"
    }

    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
$results += ""
$results += "done"
$results | Set-Content $outFile -Encoding UTF8
Write-Output "FINAL SWEEP DONE"
