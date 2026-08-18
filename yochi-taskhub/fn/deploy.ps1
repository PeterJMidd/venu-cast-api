# deploy.ps1 - build + deploy yochi-taskhub-fn (TaskHub server-side: timers + AI + admin endpoints)
# Run from this folder:  powershell -ExecutionPolicy Bypass -File .\deploy.ps1
# Uses `py -m azure.cli` (az is not on PATH). Vendored deploy, remote build OFF.

$ErrorActionPreference = "Stop"
$SUB  = "65555b10-955d-4892-a97d-4ca3f50afbb1"
$RG   = "rg-yochi-ops"
$PLAN = "yochi-ops-plan"
$APP  = "yochi-taskhub-fn"
$STORAGE = "yochiopsstorage"

Write-Host "== subscription ==" -ForegroundColor Cyan
py -m azure.cli account set --subscription $SUB

# --- 1. Create the function app if it does not yet exist (idempotent) ---
$ErrorActionPreference = "Continue"
$exists = py -m azure.cli functionapp list -g $RG --query "[?name=='$APP'].name" -o tsv
$ErrorActionPreference = "Stop"
if (-not $exists) {
    Write-Host "== creating function app $APP ==" -ForegroundColor Cyan
    py -m azure.cli functionapp create --name $APP --resource-group $RG --plan $PLAN `
        --runtime python --runtime-version 3.11 --functions-version 4 --storage-account $STORAGE
    py -m azure.cli functionapp config set --name $APP --resource-group $RG --always-on true
} else {
    Write-Host "function app $APP already exists - skipping create" -ForegroundColor Yellow
}

# --- 2. Non-secret app settings ---
Write-Host "== app settings (non-secret) ==" -ForegroundColor Cyan
py -m azure.cli functionapp config appsettings set --name $APP --resource-group $RG --settings `
    FUNCTIONS_WORKER_RUNTIME=python `
    AzureWebJobsFeatureFlags=EnableWorkerIndexing `
    SCM_DO_BUILD_DURING_DEPLOYMENT=false `
    ENABLE_ORYX_BUILD=false `
    WEBSITE_TIME_ZONE=Australia/Sydney `
    SUPABASE_URL=https://kesjkyofzdpjioafhuuw.supabase.co `
    ANTHROPIC_MODEL=claude-sonnet-4-6 `
    ANTHROPIC_MODEL_CHEAP=claude-haiku-4-5-20251001 `
    EMAIL_ENABLED=true `
    APP_URL=https://jolly-flower-042897300.7.azurestaticapps.net `
    | Out-Null
# NOTE: went live 2026-07-26 — EMAIL_OVERRIDE_TO must stay UNSET (mail goes to
# real recipients) and APP_URL is the production SWA. Do not re-add the
# localhost/override values here; they clobber live settings on every deploy.

# --- 3. Secrets: copied from sibling apps so values never live in this script ---
Write-Host "== app settings (secrets, copied from yochi-daily-insights) ==" -ForegroundColor Cyan
$anthKey  = py -m azure.cli functionapp config appsettings list -n yochi-daily-insights -g $RG --query "[?name=='ANTHROPIC_API_KEY'].value" -o tsv
$clientId = py -m azure.cli functionapp config appsettings list -n yochi-daily-insights -g $RG --query "[?name=='CLIENT_ID'].value" -o tsv
$clientSc = py -m azure.cli functionapp config appsettings list -n yochi-daily-insights -g $RG --query "[?name=='CLIENT_SECRET'].value" -o tsv
$tenantId = py -m azure.cli functionapp config appsettings list -n yochi-daily-insights -g $RG --query "[?name=='TENANT_ID'].value" -o tsv
$sender   = py -m azure.cli functionapp config appsettings list -n yochi-daily-insights -g $RG --query "[?name=='SENDER_EMAIL'].value" -o tsv
$blobCs   = py -m azure.cli functionapp config appsettings list -n yochi-daily-insights -g $RG --query "[?name=='BLOB_CONNECTION_STRING'].value" -o tsv
py -m azure.cli functionapp config appsettings set --name $APP --resource-group $RG --settings `
    ANTHROPIC_API_KEY=$anthKey `
    CLIENT_ID=$clientId `
    CLIENT_SECRET=$clientSc `
    TENANT_ID=$tenantId `
    SENDER_EMAIL=$sender `
    BLOB_CONNECTION_STRING=$blobCs `
    | Out-Null

# SUPABASE_SERVICE_ROLE_KEY cannot be auto-copied (lives only in the Supabase dashboard).
$hasKey = py -m azure.cli functionapp config appsettings list -n $APP -g $RG --query "[?name=='SUPABASE_SERVICE_ROLE_KEY'].name" -o tsv
if (-not $hasKey) {
    Write-Host "!! SUPABASE_SERVICE_ROLE_KEY is NOT set. Set it once with:" -ForegroundColor Red
    Write-Host "   py -m azure.cli functionapp config appsettings set -n $APP -g $RG --settings SUPABASE_SERVICE_ROLE_KEY=<key from Supabase dashboard -> Settings -> API keys>" -ForegroundColor Red
}

# --- 4. CORS for the SPA ---
Write-Host "== CORS ==" -ForegroundColor Cyan
$ErrorActionPreference = "Continue"
py -m azure.cli functionapp cors add -n $APP -g $RG --allowed-origins http://localhost:3100 2>$null | Out-Null
$ErrorActionPreference = "Stop"

# --- 5. Vendor dependencies (Linux wheels) ---
if (-not (Test-Path ".python_packages\lib\site-packages\jwt")) {
    Write-Host "== vendoring dependencies ==" -ForegroundColor Cyan
    py -m pip install --target .python_packages\lib\site-packages `
        --platform manylinux2014_x86_64 --implementation cp `
        --python-version 3.11 --abi cp311 --only-binary=:all: -r requirements.txt
}

# --- 6. Zip + deploy (remote build off) ---
# NOTE: Compress-Archive (PS 5.1) writes backslash entry names, which breaks
# Kudu's Linux sync with EINVAL. Zip via python (always forward slashes).
Write-Host "== zipping ==" -ForegroundColor Cyan
if (Test-Path deploy_local.zip) { Remove-Item deploy_local.zip -Force }
py -c @"
import os, zipfile
exclude = {'deploy_local.zip', 'deploy.ps1'}
with zipfile.ZipFile('deploy_local.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, '.')
            if rel in exclude or '__pycache__' in rel:
                continue
            z.write(p, rel.replace(os.sep, '/'))
print('zipped', os.path.getsize('deploy_local.zip'), 'bytes')
"@

Write-Host "== deploying ==" -ForegroundColor Cyan
# az writes an informational WARNING to stderr here. Under PS 5.1 that becomes a
# NativeCommandError and, with ErrorActionPreference=Stop, aborts the script -
# but only when the caller redirects/pipes output. Relax it around this call so
# the deploy behaves the same however it is invoked, then check the exit code.
$ErrorActionPreference = "Continue"
py -m azure.cli functionapp deployment source config-zip -g $RG -n $APP --src deploy_local.zip --timeout 600
$zipExit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($zipExit -ne 0) { throw "config-zip deployment failed (exit $zipExit)" }

# --- 7. Restart (workers can serve stale module code after config-zip) ---
Write-Host "== restarting ==" -ForegroundColor Cyan
py -m azure.cli functionapp restart -n $APP -g $RG
Start-Sleep -Seconds 45

# --- 8. Smoke test: every DB column + lake query the code actually uses ---
if ($env:SKIP_SMOKE -eq "1") {
    Write-Host "== smoke test SKIPPED (SKIP_SMOKE=1) ==" -ForegroundColor Yellow
} else {
    Write-Host "== smoke test ==" -ForegroundColor Cyan
    $key = py -m azure.cli functionapp keys list -n $APP -g $RG --query masterKey -o tsv
    # 'quick' fits inside Azure's ~230s HTTP limit; run the full suite locally
    # (py smoke_test.py) before deploying - that is the fast feedback loop.
    $env:SMOKE_URL = "https://$APP.azurewebsites.net/api/ops_smoke?level=quick&code=$key"
    py -c @"
import json, os, time, urllib.request, urllib.error
url = os.environ['SMOKE_URL']
for attempt in range(4):
    try:
        with urllib.request.urlopen(url, timeout=240) as r:
            out = json.loads(r.read().decode())
        break
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors='replace')
        try:
            out = json.loads(body)
            break
        except Exception:
            print('  HTTP %s (attempt %d)' % (e.code, attempt + 1))
            if attempt == 3:
                raise SystemExit('smoke test unreachable')
            time.sleep(20)
    except Exception as e:
        print('  %s (attempt %d)' % (str(e)[:120], attempt + 1))
        if attempt == 3:
            raise SystemExit('smoke test unreachable')
        time.sleep(20)
print('  ' + out.get('summary', '?'))
for f in out.get('failed', []):
    print('  FAIL %-34s %s' % (f['check'], f['error'][:150].replace('\n', ' ')))
raise SystemExit(0 if out.get('ok') else 1)
"@
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!! SMOKE TEST FAILED - the deploy is live but something is broken." -ForegroundColor Red
        Write-Host "   Fix and redeploy, or roll back." -ForegroundColor Red
        exit 1
    }
    Write-Host "smoke test passed" -ForegroundColor Green
}

Write-Host "== done. verify with: py -m azure.cli functionapp function list -n $APP -g $RG ==" -ForegroundColor Green
