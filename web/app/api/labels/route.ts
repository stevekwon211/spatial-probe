import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

// Local-only label sink for the H3 labeling view. Writes the FULL current verdict set (idempotent
// re-save) to experiments/occquery_v0/labels/<pool>.jsonl, one JSON record per line. The git commit
// of that file is the real seal (timestamped, HARKing-visible); this route just persists from the
// browser during `npm run dev`. No-op meaningfully on a static/Vercel deploy (labeling is local work).
export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { pool_id?: string; verdicts?: unknown[] };
    const poolId = String(body.pool_id ?? "").replace(/[^a-z0-9_-]/gi, "");
    if (!poolId) return NextResponse.json({ error: "missing pool_id" }, { status: 400 });
    const verdicts = Array.isArray(body.verdicts) ? body.verdicts : [];
    const dir = path.join(process.cwd(), "..", "experiments", "occquery_v0", "labels");
    await fs.mkdir(dir, { recursive: true });
    const file = path.join(dir, `${poolId}.jsonl`);
    const lines = verdicts.map((v) => JSON.stringify(v)).join("\n");
    await fs.writeFile(file, verdicts.length ? lines + "\n" : "");
    return NextResponse.json({ written: verdicts.length, path: `experiments/occquery_v0/labels/${poolId}.jsonl` });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
