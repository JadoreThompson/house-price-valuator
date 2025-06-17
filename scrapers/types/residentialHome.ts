interface ResidentialHome {
  uprn: string;
  address: Address;

  soldDate: Date;
  priceStr: string;
  price: number;
  features: ResidentialHomeFeatures;
  
  crime: Crime;
  floodRisk: string; // E.g. very low, low, high, very high
  nearbyPOIs: PointOfInterst[];
}
