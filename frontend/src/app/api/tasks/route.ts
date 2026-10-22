import { NextResponse } from "next/server";

import { createProxyResponse, fetchBackend } from "@/server/backend-api";
import { getServerSession } from "@/server/session";

export async function GET(request: Request) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const incoming = new URL(request.url);
  const upstreamPath = `/tasks/${incoming.search}`;

  const upstream = await fetchBackend(upstreamPath, {
    method: "GET",
    userId: session.user.id,
    cache: "no-store",
  });

  return createProxyResponse(upstream);
}
