import * as fs from "fs";
import * as path from "path";
import { PageWithCursor, connect } from "puppeteer-real-browser";
import {
  ZOOPLA_API_KEY,
  ZOOPLA_BASE_URL,
  ZOOPLA_BEARER,
  ZOOPLA_POI_ENDPOINT,
} from "./config";
import { getLatLng } from "./utils/geoUtils";
import { fetchCrimeRate, sleep } from "./utils/utils";

class ZooplaScraper {
  private readonly name: string;
  private readonly url: string;
  private readonly folder: string;

  constructor(url: string, folder: string, name: string = "zoopla") {
    this.name = name;
    this.url = url;
    this.folder = path.join(folder, this.name);
  }

  public async run(): Promise<void> {
    if (!fs.existsSync(this.folder)) {
      fs.mkdirSync(this.folder, { recursive: true });
    }

    const { page, browser } = await connect({
      headless: false,
      turnstile: true,
    });

    let currentPage = page;

    await page.goto(this.url);

    const acceptCookieBtn = await page.locator("text/Accept all");
    await acceptCookieBtn.click();

    while (true) {
      await this.scrapePages(browser, currentPage);

      try {
        const oldPage = currentPage;

        const newPagePromise = new Promise((resolve) =>
          browser.once("targetcreated", (target) => resolve(target.page()))
        );

        await currentPage.keyboard.down("Control");
        await currentPage.click("text/Next");
        await currentPage.keyboard.up("Control");

        await newPagePromise;

        const pages = await browser.pages();
        currentPage = pages[pages.length - 1] as PageWithCursor;
        await currentPage.bringToFront();
        await oldPage.close();
      } catch (error) {
        break;
      }
    }

    console.log(`${this.name} finished scraping`);
  }

  /**
   * Fetches Points of Interest (POIs) data for a given property UPRN.
   * @param {PageWithCursor} page - Puppeteer browser page with cursor control.
   * @param {string} uprn - Unique Property Reference Number for the property.
   * @returns {Promise<PointOfInterst[] | null>} A list of POIs or null if data retrieval fails.
   */
  private async fetchPOIs(
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
   * @param {PageWithCursor} page - Page of a sold home.
   * @returns {Partial<ResidentialHomeFeatures> | null} Null if necessary element(s) weren't present. Otherwise
   * returns values for {"propertyType", "bedrooms", "bathrooms", "receptions"}.
   */
  private async scrapeFeatures(page: PageWithCursor) {
    const FEATURE_QS = "._1pbf8i51";
    const FEATURE_TITLES = [
      "propertyType",
      "bedrooms",
      "bathrooms",
      "receptions",
    ] as const;

    try {
      const featuresParentContainer = await page.waitForSelector(FEATURE_QS);

      const featureContainers = await featuresParentContainer!.$$(
        "._1pbf8i56._194zg6t9"
      );

      const features = await Promise.all(
        featureContainers.map((contianer) =>
          contianer.evaluate((el) => el.textContent)
        )
      );

      let data: any = {};

      features.forEach((feat) => {
        if (feat?.includes("bed")) {
          data["bedrooms"] = feat;
        } else if (feat?.includes("bath")) {
          data["bathrooms"] = feat;
        } else if (feat?.includes("reception")) {
          data["receptions"] = feat;
        } else {
          data["propertyType"] = feat;
        }
      });

      return data as Partial<ResidentialHomeFeatures>;
    } catch (error) {
      return null;
    }
  }

  /**
   *
   * @param {PageWithCursor} page - Page of a sold home.
   * @returns {Partial<ResidentialHomeFeatures> | null} Null if necessary element(s) weren't present. Otherwise
   * returns values for {"tenure", "sqm", "epcRating"}.
   */
  private async scrapeDetails(
    page: PageWithCursor
  ): Promise<Partial<ResidentialHomeFeatures> | null> {
    const DETAILS_QS = ".agepcz0";

    try {
      const detailsParentContainer = await page.waitForSelector(DETAILS_QS);

      const detailContainers = await detailsParentContainer!.$$(
        ".jc64990.jc64994._194zg6tb"
      );
      const details = await Promise.all(
        detailContainers.map((container) =>
          container.evaluate((el) => el.textContent)
        )
      );

      let data: any = {};

      details.forEach((val) => {
        if (val?.includes("sqm")) {
          data["sqm"] = val;
        } else if (val?.includes("EPC")) {
          data["epcRating"] = val;
        } else {
          data["tenure"] = val;
        }
      });

      return data as Partial<ResidentialHomeFeatures>;
    } catch (error) {
      return null;
    }
  }

  private exists(uprn: string): boolean {
    return fs.existsSync(path.join(this.folder, `${uprn}-s1.json`));
  }

  /**
   * Persists object to the filesystem using its UPRN as the filename.
   * If a file with the same UPRN already exists, the function exits without writing.
   * After writing, the file is renamed from `s0` to `s1` to signal it's ready for reading.
   * @param {ResidentialHome} data - The residential property data to be saved.
   */
  private persist(data: ResidentialHome): void {
    const uprn = data.uprn;

    if (this.exists(uprn)) return;

    const stage0FPath = path.join(this.folder, `${uprn}-s0.json`);
    const stage1FPath = stage0FPath.replace("s0", "s1");

    fs.writeFileSync(stage0FPath, JSON.stringify(data));
    fs.renameSync(stage0FPath, stage1FPath);
  }

  /**
   * Scrapes a single property page to extract address and POIs.
   * @param {PageWithCursor} page - Puppeteer page to scrape.
   */
  private async scrapePage(
    page: PageWithCursor
  ): Promise<ResidentialHome | null> {
    const ADDRESS_QS = "._194zg6td._194zg6t6";

    let pageData: any = {};

    const details = await this.scrapeDetails(page);
    if (details) {
      const features = await this.scrapeFeatures(page);
      pageData["features"] = {
        ...details,
        ...features,
      } as ResidentialHomeFeatures;
    }

    const addressEl = await page.waitForSelector(ADDRESS_QS);
    if (!addressEl) return null;

    const addr = (await addressEl.evaluate((el) => el.textContent))!;
    const [, , postcode] = addr.split(",");

    const addrObj: Address = {
      address: addr,
      ...(await getLatLng(postcode)),
    };

    const pathParts: string[] = await page.evaluate(() =>
      window.location.pathname.split("/")
    );

    const uprn: string = pathParts[pathParts.length - 2];

    const pois: PointOfInterst[] | null = await this.fetchPOIs(page, uprn);

    pageData["uprn"] = uprn;
    pageData["address"] = addrObj;
    pageData["nearbyPOIs"] = pois;

    return pageData as ResidentialHome;
  }

  private async scrapeCard(card: any, browser: any): Promise<void> {
    const anchors = await card.$$("a");
    if (!anchors.length) return;

    const href = await anchors[0].evaluate((el: any) =>
      el.getAttribute("href")
    );

    const pathParts: string[] = href.split("/");
    const uprn: string = pathParts[pathParts.length - 2];
    if (!href || this.exists(uprn)) return;

    const soldDateElement = await card.$("._194zg6t7 time");
    const soldDateStr = await soldDateElement?.evaluate(
      (el: any) => el.textContent
    );
    if (!soldDateStr) return;

    const soldDate = new Date(
      (new Date(soldDateStr).getTime() / 1000 + 60 * 60 * 24) * 1000
    );

    const soldPriceElement = await card.$$("._1i39aq49 ._194zg6t7");
    if (soldPriceElement.length < 2) return;

    const soldPriceStr = await soldPriceElement[1].evaluate(
      (el: any) => el.textContent
    );
    if (!soldPriceStr) return;

    const soldPrice = Number.parseInt(
      soldPriceStr.replace("£", "").replace(/,/g, "")
    );

    const page_: PageWithCursor = await browser.newPage();
    await page_.goto(ZOOPLA_BASE_URL! + href);

    const data: ResidentialHome | null = await this.scrapePage(page_);

    if (data) {
      data.soldDate = soldDate;
      data.priceStr = soldPriceStr;
      data.price = soldPrice;
      data.crime = await fetchCrimeRate(
        data.address.lat,
        data.address.lng,
        data.soldDate
      );

      this.persist(data);
    }

    await page_.close();
  }

  /**
   * Iterates over property listings and scrapes each individual page.
   *
   * @param {any} browser - Puppeteer browser instance.
   * @param {PageWithCursor} page - Puppeteer page containing property listing.
   * @returns {Promise<void>}
   */
  private async scrapePages(browser: any, page: PageWithCursor): Promise<void> {
    const cards = await page.$$("[data-testid='result-item']");

    for (const card of cards) {
      const sleepDuration = Math.round(Math.random() * 1000);
      await sleep(sleepDuration);
      await this.scrapeCard(card, browser);
      await sleep(sleepDuration);
      await page.evaluate(() => window.scrollBy(0, 250));
      await sleep(sleepDuration);
    }
  }
}

export default ZooplaScraper;
