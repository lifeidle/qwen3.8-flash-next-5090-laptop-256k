# Sample GPU utilization during inference (ASCII only)
$samples = @()
$samples += "time,util_pct,mem_used,power_w,sm_clk"
for ($i = 0; $i -lt 50; $i++) {
    $ts = Get-Date -Format "HH:mm:ss"
    $line = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm --format=csv,noheader 2>&1 | Out-String).Trim()
    $samples += "$ts,$line"
    Start-Sleep -Milliseconds 1500
}
$samples | Set-Content "C:\Users\chenhua\Desktop\1\gpu_log.csv" -Encoding UTF8
Write-Output "sampling done"