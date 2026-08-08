# deploy.ps1 - build + deploy yochi-data-lake (nightly dataSights -> Parquet lake sync)
# Run from this folder:  powershell -ExecutionPolicy Bypass -File .\deploy.ps1
# Uses `py -m azure.cli` (az is not on PATH). Vendored deploy, remote build OFF.

$ErrorActionPreference = "Stop"
$SUB  = "65555b10-955d-4892-a97d-4ca3f50afbb1"
$RG   = "rg-yochi-ops"
$PLAN = "yochi-ops-plan"
$APP  = "yochi-data-lake"
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
    LAKE_CRON="0 45 4 * * *" `
    TOPUP_CRON="0 30 10 * * *" `
    LAKE_CONTAINER=datasights-lake `
    REFRESH_MONTHS=2 `
    DS_SQL_SERVER=yourdatahubyochi.database.windows.net `
    DS_SQL_DB=yourdatahub_yochi `
    DS_SQL_USER=yochiClientUser `
    | Out-Null

# --- 3. Secrets: copied from the v2 app so values never live in this script ---
Write-Host "== app settings (secrets, copied from v2) ==" -ForegroundColor Cyan
$dsPass = py -m azure.cli functionapp config appsettings list -n yochi-datasights-report-clean-v2 -g $RG --query "[?name=='DS_SQL_PASSWORD'].value" -o tsv
$blobCs = py -m azure.cli functionapp config appsettings list -n yochi-daily-insights -g $RG --query "[?name=='BLOB_CONNECTION_STRING'].value" -o tsv
py -m azure.cli functionapp config appsettings set --name $APP --resource-group $RG --settings `
    DS_SQL_PASSWORD=$dsPass `
    BLOB_CONNECTION_STRING=$blobCs `
    | Out-Null

# --- 4. Vendor dependencies (Linux wheels) ---
if (-not (Test-Path ".python_packages\lib\site-packages\pymssql")) {
    Write-Host "== vendoring dependencies ==" -ForegroundColor Cyan
    py -m pip install --target .python_packages\lib\site-packages `
        --platform manylinux2014_x86_64 --implementation cp `
        --python-version 3.11 --abi cp311 --only-binary=:all: -r requirements.txt
}

# --- 5. Zip + deploy (remote build off) ---
# Build the zip with Python's zipfile using FORWARD-SLASH arcnames. PowerShell's
# Compress-Archive writes backslash path separators that Kudu's Linux extractor
# intermittently fails on ("Zip deployment failed ... Extract zip", status 3).
Write-Host "== zipping (python zipfile, forward slashes) ==" -ForegroundColor Cyan
py build_zip.py

Write-Host "== deploying ==" -ForegroundColor Cyan
py -m azure.cli functionapp deployment source config-zip -g $RG -n $APP --src deploy_local.zip --timeout 900

Write-Host "== done. verify with: py -m azure.cli functionapp function list -n $APP -g $RG ==" -ForegroundColor Green
