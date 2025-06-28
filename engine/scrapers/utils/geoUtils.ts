import NodeGeocoder from "node-geocoder";
import { LOCATION_API_KEY } from "../config";
import { sleep } from "./utils";

const LAT_LNG_RATE_LIMIT: number = 10;
let lastLatLngCall: number = 0;

const geocoder = NodeGeocoder({
  provider: "locationiq",
  apiKey: LOCATION_API_KEY,
});

export class LocatioNotFound extends Error {}

/**
 * Gets the latitude and longitude for a given postcode
 * and applies a basic rate limit.
 *
 * @param {string} postcode - The postcode to geocode.
 * @returns {Promise<{ lat: number; lng: number }>} An object containing latitude and longitude.
 * @throws {LocationNotFoundError} If no location is found for the given postcode.
 */
export async function getLatLng(
  postcode: string
): Promise<{ lat: number; lng: number }> {
  const sleepTime: number = Math.max(
    0,
    lastLatLngCall + LAT_LNG_RATE_LIMIT - Date.now()
  );

  if (sleepTime) {
    await sleep(sleepTime);
  }

  const res = await geocoder.geocode(postcode);

  if (res.length) {
    const { latitude, longitude } = res[0];
    lastLatLngCall = Date.now();
    return {
      lat: latitude as number,
      lng: longitude as number,
    };
  }

  throw new LocatioNotFound("Location not found");
}
