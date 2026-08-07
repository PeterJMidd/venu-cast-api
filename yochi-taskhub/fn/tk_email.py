# -*- coding: utf-8 -*-
"""Microsoft Graph sendMail helper (copied from yochi-daily-insights) with the
standard kill-switch: EMAIL_ENABLED=false by default, EMAIL_OVERRIDE_TO to force
every mail to one inbox during testing."""
import os
import json
import logging
import urllib.request
import urllib.parse

LOG = logging.getLogger("tk_email")
GRAPH = "https://graph.microsoft.com/v1.0"


def _token():
    data = urllib.parse.urlencode({
        "client_id": os.environ["CLIENT_ID"],
        "client_secret": os.environ["CLIENT_SECRET"],
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }).encode()
    req = urllib.request.Request(
        "https://login.microsoftonline.com/%s/oauth2/v2.0/token" % os.environ["TENANT_ID"],
        data=data)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())["access_token"]


def enabled():
    return os.environ.get("EMAIL_ENABLED", "false").lower() == "true"


def send(to_addr, subject, html, attachments=None):
    """attachments: list of {name, data (bytes), mime}."""
    if not enabled():
        LOG.info("EMAIL_ENABLED=false - skipping email to %s (%s)", to_addr, subject)
        return False
    override = os.environ.get("EMAIL_OVERRIDE_TO", "").strip()
    if override:
        subject = "[TEST -> %s] %s" % (to_addr, subject)
        to_addr = override
    sender = os.environ.get("SENDER_EMAIL") or "peterm@yochi.com.au"
    tok = _token()
    message = {
        "subject": subject,
        "body": {"contentType": "HTML", "content": html},
        "toRecipients": [{"emailAddress": {"address": to_addr}}],
    }
    if attachments:
        import base64
        message["attachments"] = [{
            "@odata.type": "#microsoft.graph.fileAttachment",
            "name": a["name"],
            "contentType": a.get("mime", "application/octet-stream"),
            "contentBytes": base64.b64encode(a["data"]).decode(),
        } for a in attachments]
    msg = {"message": message, "saveToSentItems": True}
    req = urllib.request.Request(
        "%s/users/%s/sendMail" % (GRAPH, urllib.parse.quote(sender)),
        data=json.dumps(msg).encode(),
        headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"},
        method="POST")
    urllib.request.urlopen(req, timeout=30).read()
    LOG.info("email sent to %s", to_addr)
    return True
