"""Unmix desktop shell: runs the FastAPI server in-process and shows it in a native window."""
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import webview

ROOT = Path(__file__).parent
STATE = Path.home() / "Library" / "Application Support" / "Unmix" if getattr(sys, "frozen", False) else ROOT  # keep in sync with app.py
sys.path.insert(0, str(ROOT))
PORT = int(os.environ.get("UNMIX_PORT", 8765))
URL = f"http://127.0.0.1:{PORT}"
ICON = ROOT / "assets" / "Unmix.icns"

LOADING = """<body style="margin:0;height:100vh;display:grid;place-items:center;background:#111;color:#aaa;
font:15px -apple-system,system-ui">Starting Unmix…</body>"""


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def wait_ready(timeout=120) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(f"{URL}/api/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.2)
    return False


def brand():
    """Show the Unmix name and icon in the Dock / menu bar instead of Python's."""
    try:
        from AppKit import NSApplication, NSImage
        from Foundation import NSBundle

        info = NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = "Unmix"
        NSApplication.sharedApplication().setApplicationIconImage_(
            NSImage.alloc().initWithContentsOfFile_(str(ICON)))
    except Exception:
        pass


def open_files_from_finder():
    """Double-clicking a .umx file in Finder: queue it for the app to open (works whether Unmix is running or not)."""
    try:
        import objc
        import webview.platforms.cocoa as cocoa

        import openq

        class AppDelegate(objc.Category(cocoa.BrowserView.AppDelegate)):
            def application_openFiles_(self, app, files):
                openq.paths.extend(str(f) for f in files)
                app.replyToOpenOrPrint_(0)  # NSApplicationDelegateReplySuccess
                app.activateIgnoringOtherApps_(True)
    except Exception:
        pass


def single_instance():
    """Hold an exclusive lock for the life of the process; if another Unmix has it, focus that one and quit."""
    import fcntl
    import os

    lock_dir = STATE / "data"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = open(lock_dir / ".unmix.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.seek(0)
        try:
            from AppKit import NSRunningApplication

            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(lock.read().strip()))
            if app:
                app.activateWithOptions_(1 << 1)  # NSApplicationActivateIgnoringOtherApps
        except Exception:
            pass
        sys.exit(0)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    return lock  # keep referenced


def main():
    _lock = single_instance()
    open_files_from_finder()
    existing = port_open()
    window = webview.create_window("Unmix", URL if existing else None, html=None if existing else LOADING,
                                   width=1280, height=860, min_size=(900, 640))
    state = {}

    def boot():
        brand()
        if existing:
            return
        import uvicorn

        import app as unmix  # heavy imports; runs while the loading screen is visible
        state["server"] = uvicorn.Server(uvicorn.Config(unmix.app, host="127.0.0.1", port=PORT, log_level="warning"))
        threading.Thread(target=state["server"].run, daemon=True).start()
        if wait_ready():
            window.load_url(URL)

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.start(boot)
    if "server" in state:
        state["server"].should_exit = True


if __name__ == "__main__":
    main()
