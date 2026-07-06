import { redirect } from "next/navigation";

// Free-Space is no longer a page — the geometry is the "Geometry" overlay inside the one Explorer.
export default function FreeSpacePage() {
  redirect("/occquery");
}
