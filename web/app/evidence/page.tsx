import Link from "next/link";
import { ChevronLeft } from "lucide-react";
import { EvidenceLedger } from "@/components/probe/evidence-ledger";

export const metadata = {
  title: "Evidence — spatial-probe",
  description: "Every claim, its verdict, and how it was graded. Negatives included.",
};

export default function EvidencePage() {
  return (
    <div className="min-h-screen bg-[#080808]">
      <div className="absolute top-4 left-4 z-10">
        <Link
          href="/"
          className="flex items-center gap-1 text-sm font-medium tracking-tight text-white/80 transition-colors hover:text-white"
        >
          <ChevronLeft className="size-3.5 text-white/40" />
          spatial-probe
        </Link>
      </div>
      <EvidenceLedger />
    </div>
  );
}
