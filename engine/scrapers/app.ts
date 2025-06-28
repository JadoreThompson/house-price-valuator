import { SCRAPED_DATA_FOLDER, ZOOPLA_SOURCE } from "./config";
import ZooplaScraper from "./zoopla";

const start = async (): Promise<void> => {
  await new ZooplaScraper(ZOOPLA_SOURCE!, SCRAPED_DATA_FOLDER).run();
};

start();
