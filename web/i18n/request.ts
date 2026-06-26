import { cookies } from "next/headers";
import { getRequestConfig } from "next-intl/server";
import { LOCALE_COOKIE, defaultLocale, isLocale } from "./config";

// Cookie / no-i18n-routing mode: the active locale comes from the NEXT_LOCALE cookie, not the URL.
// Falls back to the canonical default ('en') when the cookie is missing or holds an unknown value.
export default getRequestConfig(async () => {
  const store = await cookies();
  const cookieValue = store.get(LOCALE_COOKIE)?.value;
  const locale = isLocale(cookieValue) ? cookieValue : defaultLocale;

  return {
    locale,
    messages: (await import(`../messages/${locale}.json`)).default,
  };
});
