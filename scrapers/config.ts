import dotenv from "dotenv";
import * as path from "path";

dotenv.config();

export const SCRAPED_DATA_FOLDER = path.join(
  path.dirname(path.dirname(path.dirname(__filename))),
  "datasets",
  "raw-scraped-data"
);
export const LOCATION_API_KEY = process.env.LOCATION_API_KEY;
export const CRIME_RATE_ENDPOINT = process.env.CRIME_RATE;

export const ZOOPLA_SOURCE = process.env.ZOOPLA_SOURCE;
export const ZOOPLA_BASE_URL = process.env.ZOOPLA;
export const ZOOPLA_POI_ENDPOINT = process.env.ZOOPLA_POI_ENDPOINT;
export const ZOOPLA_BEARER = process.env.ZOOPLA_BEARER;
export const ZOOPLA_API_KEY = process.env.ZOOPLA_API_KEY;
export const ZOOPLA_FLOOD_RISK_ENDPOINT = process.env.ZOOPLA_FLOOD_RISK;
export const ZOOPLA_CRIME_ENDPOINT = process.env.ZOOPLA_CRIME;
