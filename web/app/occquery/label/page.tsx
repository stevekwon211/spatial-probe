import { Labeler } from "@/components/occquery/labeler";

// Blind H3 ground-truth labeling. The pool + queries are sealed (committed before labeling); the
// predicate's verdict is not loaded until a vote locks. See experiments/occquery_v0/preregistration.md.
export default function LabelPage() {
  return <Labeler />;
}
