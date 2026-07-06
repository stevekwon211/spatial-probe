import { Explorer } from "@/components/occquery/explorer";

export default async function OccqueryPage({ searchParams }: { searchParams: Promise<{ view?: string }> }) {
  const { view } = await searchParams;
  return <Explorer initialView={view === "geometry" ? "geometry" : "query"} />;
}
