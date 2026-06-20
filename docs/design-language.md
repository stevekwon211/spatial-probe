# occquery — design language (1-pager)

**Mission → design.** "3D is queryable *state*, not the render." So the interface must get out of the
way and let the state be seen and questioned. The product's whole job is two verbs — **query** and
**see** — and design's job is to serve both, then disappear.

## Principles — design masters, re-read for occquery

| master | their principle | our reading |
|---|---|---|
| **Dieter Rams** | "As little design as possible." | The calmest tool in the stack. Chrome yields to state. |
| **Jony Ive** | Deference; bring order to complexity. | UI floats as glass *over* the 3D; the state is the hero, not the panels. |
| **Edward Tufte** | "Above all else, show the data." Maximize data-ink. | Erase chrome-ink. The only color in the product belongs to occupancy. |
| **Kenya Hara (MUJI)** | Emptiness (空) as a vessel. | Whitespace isn't empty — it's *ready*. The void holds the next query. |
| **John Maeda** | Laws of Simplicity: reduce, then organize. | Nothing until needed; power on demand. Hide enterprise depth behind calm. |
| **Massimo Vignelli** | "Design is one." Grid + type discipline. | One grid, one typeface, one logic across every surface. Timeless over trendy. |
| **Charles & Ray Eames** | "The details are not the details — they make the design." | Hairline borders, one exact radius, honest numbers down to the last cell. |
| **Brian Chesky** | The 11-star experience. | Reviewing thousands of scenes should feel calm — even, occasionally, wondrous. |

## Visual system

- **Achromatic.** Near-black canvas (`#08`), glass surfaces (`white 4–8%` + `backdrop-blur` + hairline
  `white/10`). **No brand accent color.** Active state = `white/10`. Color is reserved for data.
- **Color belongs to data.** Only occupancy (semantic voxels) carries hue.
- **Type.** One family (Geist), three sizes. **Measurements set in mono** — a signal: *this is data*.
- **Material.** Liquid glass = the query layer floating over the state (the philosophy made physical).
- **Motion.** Functional only, near-zero. No decorative animation.
- **Icons.** lucide, thin stroke, consistent. **No emoji, ever.**
- **Geometry.** One large radius (`rounded-2xl`). One grid.

## Layout — calm by default

3D fills the canvas. Floating glass clusters appear only as needed:
- **top-left** — context (scene / frame) + a quiet entry to view controls
- **bottom-center** — time (playback); the time axis is horizontal because time is
- **right** — query / chat, *summoned*, not always on
- **left** — nav rail, expands only when moving between modules

## The one test (Rams 10)

For every element: *does it help the engineer query or see?* If not, remove it.

## Anti-patterns (what enterprise tools do — we don't)

panels everywhere → progressive disclosure · many accents → achromatic, data-only color ·
dense by default → calm by default · decorative/emoji icons → lucide, functional · trendy → timeless
