# Build deploy_local.zip with FORWARD-SLASH arcnames.
# PowerShell's Compress-Archive writes backslash separators that Kudu's Linux
# extractor intermittently fails on ("Extract zip", status 3). Run: py build_zip.py
import os
import zipfile

OUT = "deploy_local.zip"
EXC_DIRS = {"__pycache__", ".git", ".vscode", ".venv"}
EXC_FILES = {"deploy_local.zip", "deploy_flat.zip", "deploy.ps1", "build_zip.py"}

if os.path.exists(OUT):
    os.remove(OUT)

n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for dp, dns, fns in os.walk("."):
        dns[:] = [d for d in dns if d not in EXC_DIRS]
        for f in fns:
            if f in EXC_FILES or f.endswith(".pyc"):
                continue
            full = os.path.join(dp, f)
            arc = os.path.relpath(full, ".").replace(os.sep, "/")
            z.write(full, arc)
            n += 1
print("zipped %d files -> %s (%.1f MB)" % (n, OUT, os.path.getsize(OUT) / 1e6))
