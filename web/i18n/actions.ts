"use server";

import { cookies } from "next/headers";
import { LOCALE_COOKIE, type Locale, isLocale } from "./config";

// next-intl cookie pattern: the toggle calls this server action to persist the chosen locale,
// then triggers router.refresh() so the server re-renders with the new messages. No URL change.
export async function setLocale(locale: Locale): Promise<void> {
  if (!isLocale(locale)) return;
  const store = await cookies();
  store.set(LOCALE_COOKIE, locale, {
    path: "/",
    maxAge: 60 * 60 * 24 * 365, // 1 year
    sameSite: "lax",
  });
}
