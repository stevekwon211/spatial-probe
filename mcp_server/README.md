# spatial-probe MCP server

The research instrument as LLM-callable tools. The program's thesis is "3D is queryable, updatable
**state**, not the render" — this server makes that literal: an agent (Claude Code, or any MCP client)
can call a falsifiable physical predicate on a real Occ3D-nuScenes scene and get the measurement back,
instead of writing a one-off script each time.

## Tools

| tool | what it does | needs local data? |
|---|---|---|
| `list_scenes()` | available Occ3D-nuScenes scene names | yes |
| `scene_info(scene)` | frame count + per-frame ego speed | yes |
| `list_predicates()` | the runnable predicates + what each measures | no |
| `probe_scene(scene, frame, predicate, horizon=3.0, unknown_policy="free")` | run one predicate on one frame → the value | yes |
| `get_findings(experiment="occquery_v0")` | the committed, honest results summary | no |

Predicates: `lateral_clearance`, `min_free_width`, `free_along_path`, `centerline_lateral` (occupancy-
native free-space geometry), and `box_distance` (the box-only baseline, for contrast). `unknown_policy`
controls how UNOBSERVED voxels are treated (`free` / `occupied` / `ignored`).

`get_findings` / `list_predicates` work with no dataset; the scene tools need the gated Occ3D-nuScenes
data in `data/` (a free nuScenes research account; see the repo CLAUDE.md).

## Install + register

```sh
pip install -e ".[mcp]"          # adds the `mcp` SDK to the venv
```

The repo ships a project-scoped `.mcp.json`, so an MCP client opened in this repo auto-discovers the
server (it launches `.venv/bin/python mcp_server/server.py` from the repo root). To register it
explicitly with Claude Code instead:

```sh
claude mcp add spatial-probe -- .venv/bin/python mcp_server/server.py
```

(For another machine, point `command` at that venv's python, or run `pip install -e ".[mcp]"` there
first so `.venv/bin/python` resolves.)

## Example

```
list_scenes()                                            -> 850 scenes
probe_scene("scene-0061", 10, "lateral_clearance")       -> { value: 5.08, unit: "m", ... }
probe_scene("scene-0061", 10, "box_distance")            -> { value: 9.49, unit: "m" }  (box-only)
get_findings("occquery_v0")                              -> the H1/H3 results summary
```

The contrast between an occupancy predicate (e.g. `min_free_width`) and `box_distance` on the same frame
is occquery's H1 thesis in one call: the occupancy field measures box-blind free-space the box-only
baseline cannot express.
