import { redirect } from "next/navigation";

// Free-Space is no longer a separate page — it is the Geometry view of the one Explorer.
export default function FreeSpacePage() {
  redirect("/occquery?view=geometry");
}
