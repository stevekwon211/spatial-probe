import { ArrowRight, Play } from "lucide-react";

import {
  IN_PROGRESS,
  PIPELINE,
  SHIPPED,
  THESIS,
  TOTAL,
} from "@/lib/pipeline";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default function Page() {
  return (
    <main className="mx-auto w-full max-w-4xl px-4 py-8 sm:px-6 sm:py-10">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Pipeline</h1>
          <p className="mt-1 max-w-xl text-sm text-muted-foreground">
            {THESIS} Six falsifiable diagnostics, one per stage of the spatial
            data pipeline.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <span className="text-sm text-muted-foreground">
            {SHIPPED}/{TOTAL} shipped · {IN_PROGRESS} in&nbsp;progress
          </span>
          <Button disabled>
            <Play className="size-4" />
            Run full pipeline
          </Button>
        </div>
      </div>

      <div className="mt-10 space-y-4">
        {PIPELINE.map((stage, i) => (
          <div key={stage.id}>
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <span className="text-sm tabular-nums text-muted-foreground">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <div className="flex size-8 items-center justify-center rounded-md border bg-muted/40">
                    <stage.icon className="size-4 text-muted-foreground" />
                  </div>
                  <div>
                    <CardTitle className="text-base">{stage.name}</CardTitle>
                    <CardDescription>{stage.blurb}</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3 sm:grid-cols-2">
                  {stage.modules.map((m) => (
                    <div key={m.id} className="rounded-lg border bg-card p-4">
                      <div className="flex items-start justify-between gap-2">
                        <h3 className="font-medium">{m.title}</h3>
                        <Badge
                          variant={
                            m.status === "in-progress" ? "default" : "secondary"
                          }
                          className="shrink-0"
                        >
                          {m.statusLabel}
                        </Badge>
                      </div>
                      <p className="mt-0.5 text-xs uppercase tracking-wide text-muted-foreground">
                        {m.axis}
                      </p>
                      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                        {m.oneLine}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            {i < PIPELINE.length - 1 && (
              <div className="flex justify-center py-1 text-muted-foreground/60">
                <ArrowRight className="size-4 rotate-90" />
              </div>
            )}
          </div>
        ))}
      </div>

      <footer className="mt-12 text-sm text-muted-foreground">
        Interactive demos — occupancy scenes and live predicate queries — land
        in each stage as its experiment ships.{" "}
        <a
          href="https://github.com/stevekwon211/spatial-probe"
          className="underline underline-offset-4 hover:text-foreground"
        >
          source
        </a>
      </footer>
    </main>
  );
}
