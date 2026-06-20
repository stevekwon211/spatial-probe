# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Doeon Kwon
"""Build a static HTML gallery of the occquery retrieval on real Occ3D-nuScenes mini scenes, so the
results can be eyeballed in a browser. Renders nothing itself -- it reuses the PNGs produced by
viz.py (run `viz.py corridor`, `viz.py tight`, `viz.py blocked` first). Visual-agreement view, not a
scientific denotation result.

Usage: python experiments/occquery_v0/viz.py corridor && ... && python experiments/occquery_v0/gallery.py
"""
from __future__ import annotations

import html as _html
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from probe.adapters.occ3d import load_scene
from probe.grid import UnknownPolicy
from probe.query_spec import load_queries
from probe.retrieval import scene_matches

_DATA = _HERE.parents[1] / "data"
MINI = [
    "scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0757",
    "scene-0796", "scene-0916", "scene-1077", "scene-1094", "scene-1100",
]
QUERIES = {
    "corridor": ("corridor_narrows_below_vehicle_width", "free corridor narrows below the car width (0 < width < ego width)"),
    "tight": ("tight_clearance_at_speed", "side gap < 0.5 m while faster than 30 km/h"),
    "blocked": ("free_path_is_blocked", "ego straight-ahead path blocked in some frame (static; temporal -> dynfield)"),
}

_HEAD = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>occquery v0 - real Occ3D-nuScenes mini</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;max-width:1150px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.5}
h1{font-size:1.5rem;margin-bottom:.3rem} h2{margin-top:2.2rem;font-size:1.15rem}
.desc{font-weight:400;color:#666;font-size:.95rem} .note{color:#666;font-size:.85rem;margin:.2rem 0 1rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:1rem}
.card{border:1px solid #e2e2e2;border-radius:10px;padding:.4rem;margin:0;background:#fff}
.card.match{border:2px solid #16a34a;box-shadow:0 0 0 3px rgba(22,163,74,.12)}
.card img{width:100%;border-radius:6px;display:block}
figcaption{font-size:.85rem;padding:.4rem .2rem 0;color:#444}
.badge{background:#16a34a;color:#fff;font-size:.72rem;font-weight:600;padding:.1rem .5rem;border-radius:99px;margin-left:.4rem}
.intro{background:#f6f6f6;border-radius:10px;padding:.8rem 1rem;font-size:.9rem;color:#444}
code{background:#eee;padding:.05rem .3rem;border-radius:4px}
</style></head><body>
<h1>occquery v0 - predicate retrieval on real Occ3D-nuScenes mini</h1>
<div class="intro">
Top-down occupancy in the <b>ego frame</b>: red arrow = ego (driving direction is to the right),
<b>black</b> = obstacle, <b>white</b> = free space. Each tile title shows the measured values
(side clearance, free-corridor width, speed). A <b style="color:#16a34a">green border</b> means the
predicate retrieved that scene. This is a <b>visual-agreement view, not a scientific denotation
result</b> - the ground truth would be a human call on the same data (see
<code>docs/h3-real-data-findings.md</code>).
</div>
"""


def main() -> None:
    qspec = {q.id: q for q in load_queries(_HERE / "queries.yaml")}
    scenes = {s: load_scene(s, _DATA, mask="none") for s in MINI}
    parts = [_HEAD]
    for key, (qid, desc) in QUERIES.items():
        retrieved = {s for s in MINI if scene_matches(scenes[s], qspec[qid], UnknownPolicy.FREE)}
        parts.append(f"<h2>{key} <span class='desc'>- {_html.escape(desc)}</span></h2>")
        parts.append(f"<p class='note'>retrieved: {', '.join(sorted(retrieved)) or 'none'}</p>")
        parts.append("<div class='grid'>")
        for s in MINI:
            png = _HERE / "results" / "viz" / key / f"{s}.png"
            if not png.exists():
                continue
            match = s in retrieved
            badge = "<span class='badge'>MATCH</span>" if match else ""
            parts.append(
                f"<figure class='card {'match' if match else ''}'>"
                f"<img src='viz/{key}/{s}.png' loading='lazy'>"
                f"<figcaption>{s}{badge}</figcaption></figure>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    out = _HERE / "results" / "gallery.html"
    out.write_text("\n".join(parts))
    print("wrote", out)


if __name__ == "__main__":
    main()
