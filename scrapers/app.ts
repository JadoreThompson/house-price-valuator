import { connect } from "puppeteer-real-browser";
import { ZOOPLA_SOURCE } from "./config";
import scrapePages from "./zoopla";

/**
 * Entry point for the Zoopla scraper.
 *
 * @returns {Promise<void>}
 */
const start = async (): Promise<void> => {
  const { page, browser } = await connect({ headless: false, turnstile: true });

  await page.goto(ZOOPLA_SOURCE!);

  const acceptCookieBtn = page.locator("text/Accept all");
  await new Promise((resolve) => setTimeout(resolve, 10));
  await acceptCookieBtn.click();

  await scrapePages(browser, page);
};

start();
