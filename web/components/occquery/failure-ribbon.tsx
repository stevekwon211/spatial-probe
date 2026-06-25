"use client";

import { useCallback, useMemo } from "react";
import type { DeltaPoint, FrameEvent } from "./events";

// SVG layout constants
const SVG_HEIGHT = 56;
const AXIS_HEIGHT = 14; // bottom mono tick row
const SPARKLINE_HEIGHT = SVG_HEIGHT - AXIS_HEIGHT; // 42px area for the sparkline
const DIAMOND_SIZE = 5;

function Diamond({ cx, cy, color }: { cx: number; cy: number; color: string }) {
  const half = DIAMOND_SIZE / 2;
  return (
    <polygon
      points={`${cx},${cy - half} ${cx + half},${cy} ${cx},${cy + half} ${cx - half},${cy}`}
      fill={color}
    />
  );
}

type Props = {
  series: DeltaPoint[];
  events: FrameEvent[];
  frameIdx: number;
  onSeek: (idx: number) => void;
  totalFrames: number;
};

export function FailureRibbon({ series, events, frameIdx, onSeek, totalFrames }: Props) {
  const width = 480; // intrinsic SVG width; scales with viewBox

  const { minDelta, maxDelta, points, zeroY } = useMemo(() => {
    const deltas = series.map((p) => p.delta).filter((d): d is number => d !== null);
    if (deltas.length === 0) {
      return { minDelta: 0, maxDelta: 1, points: [], zeroY: SPARKLINE_HEIGHT / 2 };
    }
    const minDelta = Math.min(0, ...deltas);
    const maxDelta = Math.max(0.001, ...deltas); // at least 1mm above zero so range > 0
    const range = maxDelta - minDelta;

    const toY = (d: number) =>
      SPARKLINE_HEIGHT - ((d - minDelta) / range) * SPARKLINE_HEIGHT;

    const zeroY = toY(0);

    const points = series.map((pt, i) => {
      const x = (i / Math.max(totalFrames - 1, 1)) * width;
      const y = pt.delta !== null ? toY(pt.delta) : zeroY;
      return { x, y, delta: pt.delta };
    });

    return { minDelta, maxDelta, points, zeroY };
  }, [series, totalFrames, width]);

  // Build SVG polyline path for the area fill
  const areaPath = useMemo(() => {
    if (points.length === 0) return "";
    const linePoints = points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
    const firstX = points[0].x.toFixed(1);
    const lastX = points[points.length - 1].x.toFixed(1);
    const zy = zeroY.toFixed(1);
    return `M${firstX},${zy} L${linePoints} L${lastX},${zy} Z`;
  }, [points, zeroY]);

  // Stroke polyline (top edge of the fill)
  const strokePath = useMemo(() => {
    if (points.length === 0) return "";
    return `M${points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" L")}`;
  }, [points]);

  const handleClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      const idx = Math.round(ratio * (totalFrames - 1));
      onSeek(Math.max(0, Math.min(totalFrames - 1, idx)));
    },
    [onSeek, totalFrames],
  );

  const currentX = totalFrames > 1 ? (frameIdx / (totalFrames - 1)) * width : 0;

  return (
    <div className="w-full select-none">
      {/* label row */}
      <div className="mb-0.5 flex items-center justify-between px-1">
        <span className="font-mono text-[10px] tracking-wide text-white/30">
          box − lateral gap (m)
        </span>
        <span className="font-mono text-[10px] text-white/20">
          click to seek
        </span>
      </div>

      <svg
        viewBox={`0 0 ${width} ${SVG_HEIGHT}`}
        preserveAspectRatio="none"
        className="w-full cursor-crosshair"
        style={{ height: `${SVG_HEIGHT}px` }}
        onClick={handleClick}
        aria-label="delta series ribbon — click to seek frame"
        role="slider"
      >
        {/* zero baseline */}
        <line
          x1={0}
          y1={zeroY}
          x2={width}
          y2={zeroY}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth={1}
        />

        {/* teal area fill */}
        {areaPath && (
          <path
            d={areaPath}
            fill="var(--data)"
            fillOpacity={0.18}
          />
        )}

        {/* teal stroke */}
        {strokePath && (
          <path
            d={strokePath}
            fill="none"
            stroke="var(--data)"
            strokeWidth={1.5}
          />
        )}

        {/* event ticks */}
        {events.map((ev) => {
          const x = (ev.frameIdx / Math.max(totalFrames - 1, 1)) * width;
          const isNearMiss = ev.kind === "NEAR_MISS";
          return (
            <g key={`${ev.kind}-${ev.frameIdx}`}>
              <line
                x1={x}
                y1={0}
                x2={x}
                y2={SPARKLINE_HEIGHT}
                stroke={isNearMiss ? "var(--data)" : "rgba(255,255,255,0.35)"}
                strokeWidth={isNearMiss ? 1.5 : 1}
                strokeDasharray={isNearMiss ? undefined : "3 2"}
              />
              {/* clickable hit target */}
              <rect
                x={x - 6}
                y={0}
                width={12}
                height={SPARKLINE_HEIGHT}
                fill="transparent"
                onClick={(e) => { e.stopPropagation(); onSeek(ev.frameIdx); }}
                className="cursor-pointer"
              />
              {/* tick label below sparkline */}
              <text
                x={x}
                y={SPARKLINE_HEIGHT + 10}
                textAnchor="middle"
                fill={isNearMiss ? "var(--data-text)" : "rgba(255,255,255,0.30)"}
                fontSize={9}
                fontFamily="var(--font-mono)"
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {isNearMiss ? "NM" : "VF"}
              </text>
            </g>
          );
        })}

        {/* current frame diamond */}
        <Diamond
          cx={currentX}
          cy={points[frameIdx]?.y ?? zeroY}
          color="var(--data)"
        />

        {/* axis min/max labels */}
        <text
          x={2}
          y={SPARKLINE_HEIGHT + 12}
          fill="rgba(255,255,255,0.20)"
          fontSize={9}
          fontFamily="var(--font-mono)"
          style={{ fontVariantNumeric: "tabular-nums" }}
        >
          {minDelta.toFixed(1)}
        </text>
        <text
          x={width - 2}
          y={SPARKLINE_HEIGHT + 12}
          textAnchor="end"
          fill="rgba(255,255,255,0.20)"
          fontSize={9}
          fontFamily="var(--font-mono)"
          style={{ fontVariantNumeric: "tabular-nums" }}
        >
          {maxDelta.toFixed(1)}
        </text>
      </svg>

      {/* legend */}
      <div className="mt-0.5 flex items-center gap-3 px-1">
        <div className="flex items-center gap-1">
          <span
            className="inline-block h-0.5 w-3"
            style={{ background: "var(--data)" }}
          />
          <span className="font-mono text-[9px] text-white/25">NM = near-miss</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="font-mono text-[9px] text-white/25">- - VF = verdict flip</span>
        </div>
      </div>
    </div>
  );
}
