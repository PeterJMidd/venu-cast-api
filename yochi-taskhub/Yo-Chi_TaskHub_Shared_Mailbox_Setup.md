# Yo-Chi TaskHub — shared mailbox → task setup

**What this does:** every email arriving at a shared mailbox automatically becomes a
task in TaskHub (or attaches to the task its thread already created). Routing rules in
TaskHub (**Admin → Email routing**) decide which project the task lands in and who it is
allocated to, based on the mailbox and keywords; anything no rule claims is routed by AI.

**Mailboxes to connect:**

| Mailbox | Default routing (already set up) |
|---|---|
| finance@yochi.com.au | Finance operations |
| payroll@yochi.com.au | People & payroll |
| whsclaims@yochi.com.au | Insurance, risk & regulatory |
| solutions@yochi.com.au | AI decides per email (add a rule when a pattern emerges) |

**Who this is for:** whoever can create Power Automate flows and has (or can be granted)
access to these shared mailboxes. One flow per mailbox, ~5 minutes each. This is the same
mechanism as the existing flagged-email flow — no admin consent or new permissions needed.

---

## One flow per mailbox

1. Go to <https://make.powerautomate.com> (Yochi default environment).
2. **Create → Automated cloud flow.** Name it e.g. `TaskHub drop — finance@`.
3. Trigger: **"When a new email arrives in a shared mailbox (V2)"** (Office 365 Outlook).
   - **Mailbox Address:** `finance@yochi.com.au` (change per flow)
   - Folder: Inbox
4. Add step: **"Create file"** (SharePoint).
   - **Site Address:** Yochi Team Sharepoint
   - **Folder Path:** `Shared Documents/ALL-SHARES/Finance/SYSTEMS/TaskHub Email Drop`
   - **File Name:** use this expression (unique per message):

     ```
     concat('sharedmail-', triggerOutputs()?['body/id'], '.json')
     ```

   - **File Content:** paste this, replacing the mailbox line per flow, and insert the
     dynamic values (marked `@{...}`) from the trigger:

     ```json
     {
       "messageId": "@{triggerOutputs()?['body/internetMessageId']}",
       "mailbox": "finance@yochi.com.au",
       "subject": "@{triggerOutputs()?['body/subject']}",
       "from": "@{triggerOutputs()?['body/from']}",
       "received": "@{triggerOutputs()?['body/receivedDateTime']}",
       "webLink": "@{triggerOutputs()?['body/webLink']}",
       "body": "@{base64ToString(triggerOutputs()?['body/body']?['$content'])}"
     }
     ```

     If the body expression errors in your tenant, use the simpler
     `@{triggerOutputs()?['body/bodyPreview']}` for `"body"` — the first ~255
     characters are enough for routing; the webLink opens the full mail.

5. **Save**, then send a test email to the mailbox and check a `.json` file appears in
   the drop folder. TaskHub polls the folder **every 15 minutes**; the task appears in
   the project the routing rules choose.
6. Repeat for the other three mailboxes — only the **Mailbox Address** in the trigger
   and the `"mailbox"` line in the JSON change.

The `"mailbox"` field matters: it is how TaskHub's routing rules know which address the
mail came to, so payroll rules never fire on finance mail.

---

## How routing is decided (for whoever maintains the rules)

In TaskHub, **Admin → Email routing**:

- Rules run in **position order (lowest first)**; the first match wins.
- **Keywords** match anywhere in subject or body, comma-separated, any one is enough.
  `*` matches every email — that is the mailbox's catch-all; keep catch-alls at
  position 900 so specific keyword rules (default 100) always beat them.
- A rule sets the **project**, optionally the **default allocation** (a person) and the
  **priority**. Whatever the rule leaves blank, the AI classifier fills.
- Email no rule claims is fully AI-routed, defaulting to Finance operations when unsure.

Behaviour already built in (from the existing flagged-email pipeline):

- **Replies join their thread's task** rather than creating duplicates — a reply whose
  subject matches an open email-task attaches as a comment and notifies the assignee.
- A mail containing a TaskHub task link attaches to that exact task.
- Every task keeps a link back to the original message in Outlook.
- Duplicate protection is by message id — the same mail dropped twice creates one task.

*Questions: Peter Middleton (peterm@yochi.com.au).*
