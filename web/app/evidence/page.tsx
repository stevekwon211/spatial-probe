import Link from "next/link";
import { ChevronLeft } from "lucide-react";
import { getTranslations } from "next-intl/server";
import { EvidenceLedger } from "@/components/probe/evidence-ledger";
import { LocaleToggle } from "@/components/locale-toggle";

export async function generateMetadata() {
  const t = await getTranslations("evidence");
  return {
    title: t("metaTitle"),
    description: t("metaDescription"),
  };
}

export default async function EvidencePage() {
  const t = await getTranslations();
  return (
    <div className="min-h-screen bg-[#080808]">
      <div className="absolute top-4 left-4 z-10 flex items-center gap-3">
        <Link
          href="/"
          className="flex items-center gap-1 text-sm font-medium tracking-tight text-white/80 transition-colors hover:text-white"
        >
          <ChevronLeft className="size-3.5 text-white/40" />
          {t("brand.name")}
        </Link>
        <LocaleToggle />
      </div>
      <EvidenceLedger />
    </div>
  );
}
