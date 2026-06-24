"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

// Hairline, zebra-free, Tufte data-ink. Numeric columns are mono + tabular + right-aligned. Sortable
// client-side (no dep). Used by the evidence ledger and any future query/dataset list.
export interface Column<T> {
  key: string;
  header: string;
  align?: "left" | "right";
  sortable?: boolean;
  render?: (row: T) => React.ReactNode;
  sortValue?: (row: T) => string | number;
}

export function DataTable<T>({
  columns,
  rows,
  onRowClick,
  className,
}: {
  columns: Column<T>[];
  rows: T[];
  onRowClick?: (row: T) => void;
  className?: string;
}) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);

  const sorted = sort
    ? [...rows].sort((a, b) => {
        const col = columns.find((c) => c.key === sort.key);
        const get = (r: T) => (r as Record<string, unknown>)[sort.key];
        const va = col?.sortValue ? col.sortValue(a) : String(get(a) ?? "");
        const vb = col?.sortValue ? col.sortValue(b) : String(get(b) ?? "");
        return va < vb ? -sort.dir : va > vb ? sort.dir : 0;
      })
    : rows;

  return (
    <table className={cn("w-full border-collapse text-left", className)}>
      <thead>
        <tr className="border-b border-border">
          {columns.map((c) => (
            <th
              key={c.key}
              onClick={c.sortable ? () => setSort((s) => ({ key: c.key, dir: s?.key === c.key && s.dir === 1 ? -1 : 1 })) : undefined}
              className={cn(
                "px-3 py-2 font-mono text-[10px] font-medium uppercase tracking-wider text-muted-foreground/70",
                c.align === "right" ? "text-right" : "text-left",
                c.sortable && "cursor-pointer select-none hover:text-muted-foreground",
              )}
            >
              <span className="inline-flex items-center gap-1">
                {c.header}
                {sort?.key === c.key && (sort.dir === 1 ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />)}
              </span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody className="divide-y divide-border/60">
        {sorted.map((row, i) => (
          <tr
            key={i}
            onClick={onRowClick ? () => onRowClick(row) : undefined}
            className={cn("transition-colors", onRowClick && "cursor-pointer hover:bg-white/[0.03]")}
          >
            {columns.map((c) => (
              <td
                key={c.key}
                className={cn(
                  "px-3 py-2.5 align-top text-xs",
                  c.align === "right" && "text-right font-mono tabular-nums",
                )}
              >
                {c.render ? c.render(row) : String((row as Record<string, unknown>)[c.key] ?? "")}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
