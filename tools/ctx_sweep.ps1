# Context-size sweep: does shrinking ctx free enough VRAM to move experts to GPU?
# Measures with temperature=0 + fixed 512 tokens for strict comparability.
$model  = "D:\models\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64\Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-00001-of-00028.gguf"
$mmproj = "D:\models\Qwen3.8-Flash-Next\mmproj-Qwen3.8-Flash-Next-F16.gguf"
$mtp    = "D:\models\Qwen3.8-Flash-Next-MTP\MTP\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
$tpl    = "D:\llama.cpp\templates\qwen38-fixed.jinja"
$outFile = "C:\Users\chenhua\Desktop\1\ctx_sweep_result.txt"
$logFile = "C:\Users\chenhua\Desktop\1\ctx_sweep.log"

$configs = @(
    @{ ctx = 81920;  ncmoe = 36; tag = "c80k_moe36" },
    @{ ctx = 98304;  ncmoe = 36; tag = "c96k_moe36" },
    @{ ctx = 98304;  ncmoe = 35; tag = "c96k_moe35" },
    @{ ctx = 114688; ncmoe = 36; tag = "c112k_moe36" }
)

$results = @()
$results += "ctx sweep (temp=0, 512 tok fixed; baseline 128K/moe36 = 21.60 tok/s)"
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
        "-mm", $mmproj,
        "-md", $mtp, "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"
    )

    try {
        $proc = Start-Process -FilePath "D:\llama.cpp\llama-server.exe" -ArgumentList $argList -WindowStyle Hidden -PassThru
    } catch {
        $results += "$($cfg.tag) : Start-Process failed - $($_.Exception.Message)"
        continue
    }

    $ready = $false
    for ($i = 0; $i -lt 45; $i++) {
        try { $h = Invoke-WebRequest "http://127.0.0.1:8080/health" -TimeoutSec 3 -UseBasicParsing; if ($h.StatusCode -eq 200) { $ready = $true; break } } catch {}
        Start-Sleep -Seconds 3
    }
    if (-not $ready) {
        $results += "$($cfg.tag) : startup failed / OOM"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        continue
    }

    $vram = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>&1 | Out-String).Trim()

    # strict request: temp=0, 512 tokens
    $body = '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Please explain the history of artificial intelligence, including main schools, key breakthroughs and representative achievements."}],"max_tokens":512,"temperature":0,"reasoning_effort":"low"}'
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:8080/v1/chat/completions" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 1800
        $ct = $r.usage.completion_tokens
    } catch {
        $ct = "request failed"
    }

    Start-Sleep -Seconds 2
    $log = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
    # 正确格式: eval time =   31649.26 ms /   600 tokens (   52.84 ms per token,    18.93 tokens per second)
    $ms = [regex]::Matches($log, 'eval time =\s*([\d.]+) ms /\s*(\d+) tokens \([^)]*?,\s*([\d.]+) tokens per second')
    $acc = [regex]::Matches($log, 'draft acceptance = ([\d.]+)')
    if ($ms.Count -gt 0 -and $ms[$ms.Count-1].Groups.Count -ge 4) {
        $last = $ms[$ms.Count-1]
        $lastAcc = if ($acc.Count -gt 0) { $acc[$acc.Count-1].Groups[1].Value } else { "?" }
        $results += "$($cfg.tag) | ctx=$($cfg.ctx) moe=$($cfg.ncmoe) | decode=$($last.Groups[3].Value) t/s | out=$ct | accept=$lastAcc | VRAM=$vram"
    } else {
        # 回退：抓 tg = XX t/s
        $tg = [regex]::Matches($log, 'n_gen =\s*\d+,\s*tg =\s*([\d.]+) t/s')
        $tgval = if ($tg.Count -gt 0) { $tg[$tg.Count-1].Groups[1].Value } else { "?" }
        $results += "$($cfg.tag) | ctx=$($cfg.ctx) moe=$($cfg.ncmoe) | no eval-time (fallback tg=$tgval t/s) | out=$ct | VRAM=$vram"
    }

    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
$results += ""
$results += "done"
$results | Set-Content $outFile -Encoding UTF8
Write-Output "CTX SWEEP DONE"
