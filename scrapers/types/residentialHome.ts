interface ResidentialHome {
  uprn: string;
  address: Address;
  // address: string;

  soldDate: Date;
  priceStr: string;
  price: number;
  features: ResidentialHomeFeatures;
  
  crime: Crime;
  floodRisk: string; // E.g. very low, low, high, very high
  nearbyPOIs: PointOfInterst[];
}
