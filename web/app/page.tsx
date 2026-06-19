import { PIPELINE, THESIS } from "@/lib/pipeline";

export default function Home() {
  return (
    <main className="mx-auto max-w-5xl px-6 py-20 sm:py-28">
      <header className="mb-16">
        <p className="text-sm font-medium tracking-wide text-neutral-500">
          spatial-probe
        </p>
        <h1 className="mt-3 text-balance text-3xl font-semibold leading-tight sm:text-4xl">
          {THESIS}
        </h1>
        <p className="mt-4 max-w-2xl text-pretty leading-relaxed text-neutral-600 dark:text-neutral-400">
          An instrument for probing what a spatial representation actually stores
          — and whether that state is queryable and trustworthy enough for a
          machine to act on. One method, a falsifiable physical predicate run as a
          test, applied across the data pipeline below.
        </p>
      </header>

      <ol className="overflow-hidden rounded-xl border border-neutral-200 dark:border-neutral-800">
        {PIPELINE.map((stage, i) => (
          <li
            key={stage.id}
            className="border-t border-neutral-200 bg-white p-6 first:border-t-0 dark:border-neutral-800 dark:bg-neutral-900 sm:p-8"
          >
            <div className="flex items-baseline gap-3">
              <span className="text-sm tabular-nums text-neutral-400">
                {String(i + 1).padStart(2, "0")}
              </span>
              <h2 className="text-lg font-semibold">{stage.name}</h2>
            </div>
            <p className="mt-1 pl-8 text-sm text-neutral-500">{stage.blurb}</p>

            <div className="mt-5 grid gap-3 pl-8 sm:grid-cols-2">
              {stage.modules.map((m) => (
                <div
                  key={m.id}
                  className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800"
                >
                  <div className="flex items-center justify-between gap-2">
                    <h3 className="font-medium">{m.title}</h3>
                    <span
                      className={
                        "shrink-0 rounded-full px-2 py-0.5 text-xs " +
                        (m.status === "in-progress"
                          ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300"
                          : "bg-neutral-100 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400")
                      }
                    >
                      {m.statusLabel}
                    </span>
                  </div>
                  <p className="mt-0.5 text-xs uppercase tracking-wide text-neutral-400">
                    {m.axis}
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
                    {m.oneLine}
                  </p>
                </div>
              ))}
            </div>
          </li>
        ))}
      </ol>

      <footer className="mt-16 text-sm leading-relaxed text-neutral-500">
        Interactive demos — occupancy scenes and live predicate queries — land
        here as each experiment produces results.
        <br />
        Apache-2.0 · Doeon Kwon ·{" "}
        <a
          href="https://github.com/stevekwon211/spatial-probe"
          className="underline underline-offset-4 hover:text-neutral-700 dark:hover:text-neutral-300"
        >
          source
        </a>
      </footer>
    </main>
  );
}
