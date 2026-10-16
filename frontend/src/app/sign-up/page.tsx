import { redirect } from "next/navigation";

/** Public registration is closed for now; keep login available. */
export default function SignUpPage() {
  redirect("/sign-in");
}
