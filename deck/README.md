# Pitch deck

Source of the claude.ai Slides deck (https://claude.ai/artifact/3mLXUXYYzuKXJC7ThVGyYJ, private).
`gen.py` writes `project/deck.json` and `project/slides/*.html`; every evolution number is re-checked against
`results/` on each build (`verify_evolution()`). Refresh the user count from the live `/stats` with
`python3 gen.py --fetch-users`. Previews: `preview.py` (headless Chromium via Playwright).
Paths in `gen.py` point to the VM layout (`~/handoff/deck`); adjust `OUT` to build elsewhere.
