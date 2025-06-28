from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ResidentialHomes(Base):
    __tablename__ = "residential_homes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    uprn: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    sqm: Mapped[int] = mapped_column(Integer, nullable=False)
    bedrooms: Mapped[int] = mapped_column(Integer, nullable=False)
    bathrooms: Mapped[int] = mapped_column(Integer, nullable=False)
    receptions: Mapped[int] = mapped_column(Integer, nullable=False)

    address: Mapped[str] = mapped_column(String, nullable=False)
    postcode: Mapped[str] = mapped_column(String, nullable=False)
    incode: Mapped[str] = mapped_column(String, nullable=False)
    outcode: Mapped[str] = mapped_column(String, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lng: Mapped[float] = mapped_column(Float, nullable=False)
    town: Mapped[str] = mapped_column(String, nullable=False)
    city: Mapped[str] = mapped_column(String, nullable=False)

    # Crime statistics
    crime_violence: Mapped[int] = mapped_column(Integer, nullable=False)
    crime_anti_social: Mapped[int] = mapped_column(Integer, nullable=False)
    crime_theft: Mapped[int] = mapped_column(Integer, nullable=False)
    crime_burglary: Mapped[int] = mapped_column(Integer, nullable=False)
    crime_criminal_damage: Mapped[int] = mapped_column(Integer, nullable=False)
    crime_drugs: Mapped[int] = mapped_column(Integer, nullable=False)
    crime_weapon_possession: Mapped[int] = mapped_column(Integer, nullable=False)
    crime_public_order: Mapped[int] = mapped_column(Integer, nullable=False)
    crime_robbery: Mapped[int] = mapped_column(Integer, nullable=False)
    crime_vehicle: Mapped[int] = mapped_column(Integer, nullable=False)
    crime_other: Mapped[int] = mapped_column(Integer, nullable=False)

    # Local context
    num_pois: Mapped[int] = mapped_column(Integer, nullable=False)
    num_schools: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_distance_all: Mapped[float] = mapped_column(Float, nullable=False)
    min_distance_school: Mapped[float] = mapped_column(Float, nullable=False)

    # Derived metrics
    sqm_per_bed: Mapped[float] = mapped_column(Float, nullable=False)
    sqm_per_room: Mapped[float] = mapped_column(Float, nullable=False)
    regional_avg_price: Mapped[float] = mapped_column(Float, nullable=False)
    price_per_sqm: Mapped[float] = mapped_column(Float, nullable=False)
    regional_avg_price_per_sqm: Mapped[float] = mapped_column(Float, nullable=False)
    std: Mapped[float] = mapped_column(Float, nullable=False)

    property_type: Mapped[str] = mapped_column(String, nullable=False)
    tenure: Mapped[str] = mapped_column(String, nullable=False)
    epc_rating: Mapped[str] = mapped_column(String, nullable=False)

    sold_date: Mapped[str] = mapped_column(String, nullable=False) # Timezone-aware ISO-format date string     
    sold_price: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
