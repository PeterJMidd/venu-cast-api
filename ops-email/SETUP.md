# Yo-Chi Operations Email - Azure Function Setup Guide

## Overview
This Azure Function runs daily at 5 AM AEST, pulls shift data from SharePoint, and sends formatted operations emails to configured recipients.

---

## Step 1: Azure AD App Registration

1. Go to [Azure Portal](https://portal.azure.com) > **Azure Active Directory** > **App registrations** > **New registration**
2. Name: `YochiOpsEmail`
3. Supported account types: **Single tenant**
4. Click **Register**
5. Note down:
   - **Application (client) ID** → this is your `CLIENT_ID`
   - **Directory (tenant) ID** → this is your `TENANT_ID`

### Create Client Secret
1. In the app registration, go to **Certificates & secrets** > **New client secret**
2. Description: `ops-email-secret`, Expiry: 24 months
3. Click **Add** and copy the **Value** immediately → this is your `CLIENT_SECRET`

### Set API Permissions
1. Go to **API permissions** > **Add a permission** > **Microsoft Graph** > **Application permissions**
2. Add these permissions:
   - `Sites.Read.All` (read SharePoint lists and files)
   - `Mail.Send` (send emails on behalf of users)
3. Click **Grant admin consent for [your org]** (requires admin)

---

## Step 2: Upload the Excel Config to SharePoint

1. Open the file `OpsEmailConfig.xlsx` included in this project
2. Fill in your venues:

| Venue | SenderEmail | Recipient1 | Recipient2 | ... | Recipient12 |
|-------|------------|------------|------------|-----|-------------|
| Yo-Chi Barangaroo | operations@yochi.com.au | manager1@yochi.com.au | manager2@yochi.com.au | | |
| Yo-Chi Bondi | bondi@yochi.com.au | ... | | | |

3. Upload to SharePoint: `https://embraceyochi.sharepoint.com/sites/YochiTeamSharepoint/Shared Documents/OpsEmailConfig.xlsx`

---

## Step 3: Verify SharePoint List Column Names

The script expects these internal column names in the **Operations Update** list. Check your list and update `sharepoint_client.py` if they differ:

| Expected Column Name | Description |
|---------------------|-------------|
| `Venue` | Venue/location name |
| `Date` | Shift date |
| `ShiftLeader` | Name of shift leader |
| `Shift` | Shift type: "Day/Afternoon" or "Night" |
| `ShiftSales` | Sales amount for the shift |
| `TradeSalesFeedback` | Trade/sales feedback text |
| `GuestExperienceFeedback` | Guest experience feedback text |
| `OperationsDailyTaskFeedback` | Operations daily task feedback |
| `TeamRosteringFeedback` | Team rostering feedback text |
| `StarOfTheShift` | Star of the shift team member name |

To check column internal names:
1. Go to your SharePoint list > **Settings** (gear icon) > **List settings**
2. Click each column name to see the internal name in the URL (`Field=InternalName`)

---

## Step 4: Create the Azure Function App

### Option A: Azure Portal (easiest)
1. Go to [Azure Portal](https://portal.azure.com) > **Create a resource** > **Function App**
2. Settings:
   - **Subscription**: Your subscription
   - **Resource Group**: Create new `rg-yochi-ops` or use existing
   - **Function App name**: `yochi-ops-email` (must be globally unique)
   - **Runtime stack**: Python
   - **Version**: 3.11
   - **Region**: Australia East
   - **Plan type**: Consumption (Serverless) — essentially free for daily runs
3. Click **Review + Create** > **Create**

### Option B: Azure CLI
```bash
az login
az group create --name rg-yochi-ops --location australiaeast
az storage account create --name yochiopsstorage --location australiaeast --resource-group rg-yochi-ops --sku Standard_LRS
az functionapp create --resource-group rg-yochi-ops --consumption-plan-location australiaeast --runtime python --runtime-version 3.11 --functions-version 4 --name yochi-ops-email --os-type linux --storage-account yochiopsstorage
```

---

## Step 5: Configure App Settings

In the Azure Portal, go to your Function App > **Configuration** > **Application settings** and add:

| Setting | Value |
|---------|-------|
| `TENANT_ID` | Your Azure AD tenant ID from Step 1 |
| `CLIENT_ID` | Your app registration client ID from Step 1 |
| `CLIENT_SECRET` | Your app registration client secret from Step 1 |
| `SHAREPOINT_SITE` | `embraceyochi.sharepoint.com:/sites/YochiTeamSharepoint` |
| `SHAREPOINT_LIST_NAME` | `Operations Update` |
| `CONFIG_FILE_PATH` | `Shared Documents/OpsEmailConfig.xlsx` |
| `WEBSITE_TIME_ZONE` | `AUS Eastern Standard Time` |

Click **Save**.

---

## Step 6: Deploy the Function

### Option A: VS Code (recommended)
1. Install [Azure Functions extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-azurefunctions) for VS Code
2. Open the `ops-email` folder in VS Code
3. Click the Azure icon in the sidebar > **Deploy to Function App** > select `yochi-ops-email`

### Option B: Azure Functions Core Tools (CLI)
```bash
# Install Azure Functions Core Tools if not already installed
npm install -g azure-functions-core-tools@4

# From the ops-email directory
cd ops-email
func azure functionapp publish yochi-ops-email
```

---

## Step 7: Test

1. In Azure Portal, go to your Function App > **Functions** > `ops_email_timer`
2. Click **Code + Test** > **Test/Run** to trigger manually
3. Check **Monitor** > **Invocations** to see logs and confirm emails were sent

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Auth errors | Verify TENANT_ID, CLIENT_ID, CLIENT_SECRET are correct. Ensure admin consent was granted for API permissions. |
| "Site not found" | Check SHAREPOINT_SITE matches exactly: `embraceyochi.sharepoint.com:/sites/YochiTeamSharepoint` |
| "List not found" | Verify the list display name matches SHAREPOINT_LIST_NAME exactly |
| Column errors | Check SharePoint list column internal names match the code (see Step 3) |
| Email not sending | Ensure Mail.Send permission has admin consent. Verify the sender email in Excel config is a valid mailbox. |
| Wrong time | Confirm WEBSITE_TIME_ZONE is set to `AUS Eastern Standard Time` |

---

## Cost

Azure Functions Consumption plan includes **1 million free executions per month**. Running once daily = 30 executions/month. This will cost essentially **$0**.
