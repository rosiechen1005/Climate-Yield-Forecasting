"""
Shared paths and constants for data creation and mapping.
Set RAW_DATA_DIR to where your raw inputs live (yield CSVs, MET CSVs, ghcnd-stations.txt, county shapefile).
"""
from pathlib import Path

# Output and raw data roots
DATA_DIR = Path("data")
RAW_DATA_DIR = Path("data/raw")  # override if your raw data lives elsewhere

# Raw inputs (relative to RAW_DATA_DIR)
RAW_YIELD_DIR = RAW_DATA_DIR / "corn"           # one CSV per state, e.g. IA.csv
RAW_MET_DIR = RAW_DATA_DIR / "MET"             # met_IA.csv, met_IL.csv, ...
STATIONS_FILE = RAW_DATA_DIR / "ghcnd-stations.txt"
COUNTY_SHP_PATH = RAW_DATA_DIR / "cb_2021_us_county_500k" / "cb_2021_us_county_500k.shp"

# State FIPS codes (Corn Belt)
CORN_BELT_STATES = {
    "IA": "19", "IL": "17", "IN": "18", "KS": "20",
    "MI": "26", "MN": "27", "MO": "29", "NE": "31",
    "OH": "39", "WI": "55",
}
