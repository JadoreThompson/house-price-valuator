interface ResidentialHome {
  location: Location;
  sold: Date;
  priceStr: string;
  price: number;
  crime: Crime;
  floodRisk: string; // E.g. very low, low, high, very high
}
