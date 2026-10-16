import { SignIn } from "@/components/auth/sign-in";
import { AuthFrame } from "@/components/auth/auth-frame";

export default function SignInPage() {
  return (
    <AuthFrame footer={<>New accounts are closed for now. Sign in with an existing MartClip account.</>}>
      <SignIn />
    </AuthFrame>
  );
}
