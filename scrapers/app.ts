import { PageWithCursor, connect } from "puppeteer-real-browser";
import {
  ZOOPLA_API_KEY,
  ZOOPLA_BASE_URL,
  ZOOPLA_BEARER,
  ZOOPLA_POI_ENDPOINT,
  ZOOPLA_SOURCE,
} from "./config";
import { getLatLng } from "./utils/geoUtils";

/**
 * Fetches Points of Interest (POIs) data for a given property UPRN.
 *
 * @param {PageWithCursor} page - Puppeteer browser page with cursor control.
 * @param {string} uprn - Unique Property Reference Number for the property.
 * @returns {Promise<PointOfInterst[] | null>} A list of POIs or null if data retrieval fails.
 */
async function fetchPOIs(
  page: PageWithCursor,
  uprn: string
): Promise<PointOfInterst[] | null> {
  const data = await page.evaluate(
    (uprn, POI_ENDPOINT, BEARER, BASE_URL, API_KEY) => {
      return fetch(POI_ENDPOINT!, {
        method: "POST",
        headers: {
          accept: "application/json",
          "accept-encoding": "gzip, deflate, br, zstd",
          "accept-language": "en-GB,en;q=0.9",
          authorization: `Bearer ${BEARER}`,
          "content-length": "411",
          "content-type": "application/json",
          referer: BASE_URL!,
          "user-agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
          "x-api-key": API_KEY!,
        },
        body: JSON.stringify({
          operationName: "propertyClient",
          variables: { uprn },
          query:
            "query propertyClient($uprn:String) {breadcrumbs(uprn:$uprn){label tracking uri noAppend i18nLabelKey}property(uprn:$uprn){pointsOfInterestV2{latitude longitude distanceMiles name ...on SchoolPointOfInterest{postcode category ageLow ageHigh gender religion ofstedRating ofstedLastInspectionDate}...on TransportPointOfInterest{type}}}}",
        }),
        credentials: "include",
        mode: "cors",
      })
        .then((rsp) => rsp.json())
        .catch(() => {});
    },
    uprn,
    ZOOPLA_POI_ENDPOINT,
    ZOOPLA_BEARER,
    ZOOPLA_BASE_URL,
    ZOOPLA_API_KEY
  );

  if (!data) return null;

  return data.data.property.pointsOfInterestV2.map(
    (poi: Record<string, string | number | null>) => ({
      category: poi.category,
      distanceMiles: poi.distanceMiles,
      lat: poi.latitude,
      lng: poi.longitude,
      name: poi.name,
      gender: poi.gender,
    })
  ) as PointOfInterst[];
}

/**
 * Scrapes a single property page to extract address and POIs.
 *
 * @param {PageWithCursor} page - Puppeteer page to scrape.
 */
async function scrapePage(page: PageWithCursor) {
  const ADDRESS_QS = "._194zg6td._194zg6t6";
  const addressEl = await page.waitForSelector(ADDRESS_QS);

  if (addressEl) {
    const addr = await addressEl.evaluate((el) => el.textContent);
    let addrObj: Address;

    if (addr) {
      const [address, city, postcode] = addr.split(",");

      addrObj = {
        address: address.toLowerCase(),
        city: city.toLowerCase(),
        postcode: postcode.toLowerCase(),
        ...(await getLatLng(postcode)),
      };
    }

    const parts: string[] = await page.evaluate(() =>
      window.location.pathname.split("/")
    );
    const uprn = parts[parts.length - 2];

    const pois = await fetchPOIs(page, uprn);
  }
}

/**
 * Iterates over property listings and scrapes each individual page.
 *
 * @param {any} browser - Puppeteer browser instance.
 * @param {PageWithCursor} page - Puppeteer page containing property listing.
 * @returns {Promise<void>}
 */
async function scrapePages(browser: any, page: PageWithCursor): Promise<void> {
  const parentContainer = await page.waitForSelector(
    '[data-testid="result-item"]'
  );
  const cards = await parentContainer?.$$("div");

  for (const card of cards!) {
    const anchors = await card.$$("a");

    if (!anchors.length) continue;

    const anchor = anchors[0];
    const href = await anchor.evaluate((el) => el.getAttribute("href"));

    const page_ = await browser.newPage();
    await page_.goto(ZOOPLA_BASE_URL! + href);
    await scrapePage(page_);
  }
}

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
