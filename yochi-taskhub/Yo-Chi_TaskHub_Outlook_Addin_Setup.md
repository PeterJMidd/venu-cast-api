# Yo-Chi TaskHub — Outlook add-in setup

**What it does:** adds a **Yo-Chi TaskHub** button to Outlook. With an email open you can
either attach that email to an existing task, or create a new task from it — the subject,
sender, date and body come across automatically, and a link back to the original email is
kept on the task.

**Who this is for:** whoever installs Office add-ins for us (IT / Microsoft 365 admin), or
any individual who just wants it on their own mailbox.

---

## What you need

| | |
|---|---|
| Manifest URL | `https://jolly-flower-042897300.7.azurestaticapps.net/addin/manifest.xml` |
| Add-in name | Yo-Chi TaskHub |
| Add-in ID | `ba4f9a70-4f35-4dcc-8626-3c34f6830efe` |
| Publisher | Yo-Chi (internal — not from the Office Store) |
| Works in | Outlook on the web, new Outlook for Windows/Mac, classic Outlook (Microsoft 365) |
| Sign-in | Each person signs in with their own TaskHub email and password |

Nothing to download or install on the machine. It is a web add-in: Outlook loads it from the
URL above, so it updates itself and there is nothing to patch later.

---

## Option A — one person, their own mailbox (2 minutes)

Best for trying it out. No admin rights needed, unless our tenant blocks self-service add-ins
(see *If it doesn't appear*).

1. Open **Outlook on the web** — <https://outlook.office.com/mail/>
2. Open any email.
3. On the email's toolbar choose **… (More actions) → Get Add-ins**.
   *(Or: Settings gear → General → Manage add-ins.)*
4. Choose **My add-ins** on the left.
5. Scroll to **Custom Add-ins** at the bottom → **Add a custom add-in** → **Add from URL...**
6. Paste the manifest URL:

   ```
   https://jolly-flower-042897300.7.azurestaticapps.net/addin/manifest.xml
   ```

7. Choose **OK**, then **Install** on the warning about custom add-ins (that warning appears
   for all internally-built add-ins — it is expected).
8. Close the dialog and reopen an email. **Yo-Chi TaskHub** now appears on the toolbar,
   sometimes under **… (More actions)**.

---

## Option B — deploy for the whole team (admin, 10 minutes)

Puts it on chosen people's Outlook automatically, with nothing for them to install.

1. Sign in to the **Microsoft 365 admin centre** — <https://admin.microsoft.com>
2. Go to **Settings → Integrated apps**.
3. Choose **Upload custom apps**.
4. App type: **Office Add-in**. Then choose **Provide link to manifest file** and paste:

   ```
   https://jolly-flower-042897300.7.azurestaticapps.net/addin/manifest.xml
   ```

5. Choose who gets it: **Just me**, **Specific users/groups** (a group such as Finance is
   the usual choice), or **Entire organisation**.
6. Review the permissions — the add-in reads the currently open email only, so it can copy
   the subject and body onto the task. It does not read the rest of the mailbox.
7. Choose **Finish deployment**.

Rollout takes up to **12 hours** for existing mailboxes, and up to **24 hours** for people
who have never used add-ins. Restarting Outlook does not speed this up — it is server-side.

---

## Using it

1. Open an email in Outlook.
2. Choose **Yo-Chi TaskHub** on the toolbar (check **… More actions** if it is not visible).
3. Sign in once with your TaskHub email and password. It remembers you after that.
4. Then either:
   - **Attach to a task** — start typing a task title, pick it, add an optional note saying
     why the email matters, and choose **Attach this email**.
   - **New task** — the title, description and date are pre-filled from the email; choose the
     project and priority, then **Create task**.

Either way the task keeps a link back to the original email, so anyone opening the task can
read the source.

---

## If it doesn't appear

**"Add from URL" is missing, or the upload is refused.**
Our tenant blocks people installing their own add-ins. This is the most common outcome and it
is a policy, not a fault. Use **Option B** (admin deployment) instead — an admin can always
deploy it even when self-service is off.

**Installed but no button.**
Look under **… (More actions)** on the email toolbar — Outlook hides add-ins there when the
toolbar is full. Then fully close and reopen Outlook. On a fresh admin deployment, allow the
12–24 hours above before treating it as broken.

**"Sorry, we couldn't add the add-in" / manifest error.**
Confirm the URL opens in a browser and shows XML. If it does, the block is tenant policy —
go to Option B.

**Sign-in fails.**
The account needs an active TaskHub login. Peter can create one, or check the account is
still active, from **Admin → People** in TaskHub.

**Nothing happens when attaching.**
The add-in talks to TaskHub over the internet; a VPN or proxy that blocks
`*.azurestaticapps.net` or `*.supabase.co` will stop it. Those two domains need to be
reachable.

---

## Notes for whoever owns this afterwards

- The add-in is served from our own Azure Static Web App; **the manifest URL never changes**,
  so a redeploy of TaskHub updates the add-in for everybody with nothing to reinstall.
- It is not in the Office Store and is not meant to be — it only makes sense against our
  TaskHub.
- Removing it: individuals via the same **My add-ins** screen; org-wide via **Integrated
  apps → the app → Remove**.

*Questions: Peter Middleton (peterm@yochi.com.au).*
