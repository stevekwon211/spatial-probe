"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { OccqueryViewer } from "./viewer";
import { FreeSpaceViewer } from "@/components/freespace/viewer";

// One Explorer, two views of the same nuScenes scene: QUERY (the occupancy-predicate search demo —
// LiDAR points, boxes, verdicts, frame playback) and GEOMETRY (the honest surface — Occ3D/LiDAR mesh,
// blocky cubes, ground, projective texture, occlusion, splat). Additive wrapper: neither viewer's
// internals change; the switch just swaps the mounted body and syncs ?view= so /freespace can redirect
// here. Only the active viewer mounts (avoids loading Supabase scenes + WASM at once).
export function Explorer({ initialView }: { initialView: "query" | "geometry" }) {
  const [view, setView] = useState<"query" | "geometry">(initialView);
  const router = useRouter();
  const pick = (v: "query" | "geometry") => {
    setView(v);
    router.replace(`/occquery?view=${v}`, { scroll: false });
  };
  return (
    <div className="relative h-dvh w-full">
      <div className="absolute left-1/2 top-3 z-50 flex -translate-x-1/2 gap-1 rounded-full border border-white/10 bg-black/60 p-1 backdrop-blur">
        {(["query", "geometry"] as const).map((v) => (
          <button
            key={v}
            onClick={() => pick(v)}
            className={`rounded-full px-4 py-1.5 text-xs font-medium transition-colors ${
              view === v ? "bg-white text-black" : "text-white/60 hover:text-white"
            }`}
          >
            {v === "query" ? "Query" : "Geometry"}
          </button>
        ))}
      </div>
      {view === "geometry" ? <FreeSpaceViewer /> : <OccqueryViewer />}
    </div>
  );
}
