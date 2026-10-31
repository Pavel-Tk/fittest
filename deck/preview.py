"""Local preview: wrap slides in a 1920x1080 page and screenshot them.

Usage: python preview.py <slide_id> [...]   (no args = every slide in deck.json order)
Uses Playwright's headless Chromium if importable (on the VM: deckgen/.pwenv/bin/python preview.py ...),
otherwise falls back to a local Chrome binary (Windows laptop path).
"""
import json, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).parent
SL = HERE.parent / "deck" / "project" / "slides"
PV = HERE / "pv"
PV.mkdir(exist_ok=True)
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
HEAD = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@300..700&family=Geist+Mono:wght@400;500&display=swap">
<style>*{box-sizing:border-box;margin:0}html,body{width:1920px;height:1080px;overflow:hidden}
section{width:1920px;height:1080px;position:relative;overflow:hidden}aside{display:none}
ul,ol{padding-left:1.2em}table{border-collapse:collapse}</style></head><body>"""

ids = sys.argv[1:] or json.loads((HERE.parent / "deck" / "project" / "deck.json").read_text())["order"]
pages = []
for sid in ids:
    f = PV / f"{sid}.html"
    f.write_text(HEAD + (SL / f"{sid}.html").read_text(encoding="utf-8") + "</body></html>", encoding="utf-8")
    pages.append((f, PV / f"{sid}.png"))

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

if sync_playwright:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        for f, out in pages:
            pg.goto(f.as_uri(), wait_until="networkidle")
            pg.evaluate("document.fonts.ready")
            pg.screenshot(path=str(out))
            print(out, out.exists())
        b.close()
else:
    for f, out in pages:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1920,1080",
                        "--virtual-time-budget=4000", f"--screenshot={out}", f.as_uri()], capture_output=True, timeout=60)
        print(out, out.exists())
