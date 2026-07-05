import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";

// Scene index for the free-space view. The heavy per-scene occupancy (.occ.bin) and corridor
// (.freespace.json) are served statically from /public/occ (cacheable, streamed); this route is
// the discovery layer (matches the /api/labels handler pattern). Precompute via
// `web/scripts/prep_occ.py`.
export async function GET() {
  try {
    const p = path.join(process.cwd(), "public", "occ", "scenes.json");
    const json = JSON.parse(await readFile(p, "utf8"));
    return NextResponse.json({ available: true, ...json });
  } catch {
    return NextResponse.json({ available: false, scenes: [] });
  }
}
