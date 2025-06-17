import { CRIME_RATE_ENDPOINT } from "../config";

export async function sleep(ms: number): Promise<unknown> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * @param {string} value - The crime category string from the API.
 * @returns {keyof Crime} The corresponding key used in the `Crime` object.
 */
function parseCrimeCategory(value: string): keyof Crime {
  const map: { [key: string]: keyof Crime } = {
    "violent-crime": "violence",
    "anti-social-behaviour": "antiSocial",
    "vehicle-crime": "vehicle",
    "criminal-damage-arson": "criminalDamage",
    burglary: "burglary",
    "public-order": "publicOrder",
    drugs: "drugs",
    "other-theft": "theft",
    "theft-from-the-person": "theft",
    shoplifting: "theft",
    robbery: "robbery",
    "bicycle-theft": "bicycle",
    "possession-of-weapons": "weaponPossession",
    "other-crime": "other",
  };

  return map[value];
}

/**
 * Calculates the crime rate counts from a list of raw crime reports fetched from API.
 * @param {any[]} crimeData - Array of raw crime objects from the Police API.
 * @returns {{ [key in keyof Crime]: number }} An object with crime categories and their counts.
 */
function calculateCrimeRate(crimeData: any[]): {
  [key in keyof Crime]: number;
} {
  let crimeCounts: any = {};

  crimeData.forEach((data) => {
    const category = parseCrimeCategory(data["category"]);
    if (!crimeCounts[category]) {
      crimeCounts[category] = 0;
    }
    crimeCounts[category] += 1;
  });

  return crimeCounts as { [key in keyof Crime]: number };
}

/**
 * Fetches and processes crime data for a given location and date.
 *
 * @param {number} lat - Latitude of the target location.
 * @param {number} lng - Longitude of the target location.
 * @param {Date} date - The date for which to fetch crime statistics.
 * @returns {Promise<{ [key in keyof Crime]: number } | null>} Processed crime counts or null on failure.
 */
export async function fetchCrimeRate(
  lat: number,
  lng: number,
  date: Date
): Promise<any> {
  try {
    const rsp = await fetch(
      `${CRIME_RATE_ENDPOINT}?lat=${lat}&lng=${lng}&date=${date.getFullYear()}-${date.getMonth()}`
    );

    if (rsp.ok) return calculateCrimeRate(await rsp.json());
    throw new Error();
  } catch (error) {
    return null;
  }
}
