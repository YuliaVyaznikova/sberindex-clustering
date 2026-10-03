import re
import zipfile
from xml.etree import ElementTree

import pandas as pd
import pyogrio

from .download import FILES

RUSSIA = "Российская Федерация"
UPPER = "Муниципальное образование верхнего уровня"
BASE = ["oktmo", "oktmo_stable", "municipality", "mun_level", "year", "indicator_period", "indicator_value"]
MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
AREA_CRS = "+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m"
MUNICIPALITY_COLUMNS = {
    "territory_id": "territory_id",
    "oktmo": "oktmo",
    "municipal_district_name": "name",
    "municipal_district_type": "type",
    "municipal_district_status": "status",
    "region_code": "region_code",
    "region_name": "region",
    "year_from": "year_from",
    "year_to": "year_to",
    "municipal_district_center_lat": "lat",
    "municipal_district_center_lon": "lon",
}


def year_parts(archive, code, years):
    parts = {}
    for name in archive.namelist():
        match = re.search(rf"data_{code}_year(\d{{4}})_", name)
        if match and int(match.group(1)) in years:
            parts[int(match.group(1))] = name
    return [parts[year] for year in sorted(parts)]


def read_indicator(archive, spec):
    filters = spec.get("filters", {})
    keep = set(BASE + spec["columns"] + list(filters))
    frames = []
    for part in year_parts(archive, spec["code"], spec["years"]):
        with archive.open(part) as f:
            for chunk in pd.read_csv(f, sep=";", dtype=str, chunksize=500_000, usecols=lambda c: c in keep):
                chunk = chunk[chunk["mun_level"] == UPPER]
                for column, allowed in filters.items():
                    chunk = chunk[chunk[column].isin(allowed)]
                frames.append(chunk)
    table = pd.concat(frames, ignore_index=True)
    table = table.rename(columns={"indicator_period": "period", "indicator_value": "value"})
    return table[["oktmo", "oktmo_stable", "municipality", "year", "period", *spec["columns"], "value"]]


def prepare_rosstat(config):
    raw, out = config["paths"]["raw"], config["paths"]["prepared"]
    for name, spec in config["rosstat"].items():
        with zipfile.ZipFile(raw / FILES[spec["archive"]]) as archive:
            table = read_indicator(archive, spec)
        table.to_csv(out / f"{name}.csv.gz", index=False)
        print(f"{name}: {len(table)} rows")


def read_sdmx(path):
    root = ElementTree.parse(path).getroot()
    names = {}
    for codes in root.findall(".//{*}CodeList"):
        if codes.get("id") == "s_OKATO":
            names = {code.get("value"): code.find("{*}Description").text for code in codes.findall("{*}Code")}
    rows = []
    for series in root.findall(".//{*}Series"):
        key = {value.get("concept"): value.get("value") for value in series.findall(".//{*}Value")}
        for observation in series.findall("{*}Obs"):
            rows.append({
                "region": names[key["s_OKATO"]],
                "year": int(observation.find("{*}Time").text),
                "month": MONTHS.index(key["PERIOD"]) + 1,
                "value": float(observation.find("{*}ObsValue").get("value").replace(",", ".")),
            })
    return pd.DataFrame(rows)


def prepare_prices(config):
    prices = read_sdmx(config["paths"]["raw"] / FILES["prices"])
    mapping = config["prices"]["regions"]
    prices["preferred"] = prices["region"].isin(mapping)
    prices["region"] = prices["region"].replace(mapping)
    regions = set(pd.read_csv(config["paths"]["prepared"] / "municipalities.csv")["region"]) | {RUSSIA}
    prices = prices[prices["region"].isin(regions)]
    prices = prices.sort_values("preferred", ascending=False).drop_duplicates(["region", "year", "month"])
    prices = prices.drop(columns="preferred").sort_values(["region", "year", "month"])
    prices.to_csv(config["paths"]["prepared"] / "prices.csv", index=False)
    print(f"prices: {prices['region'].nunique() - 1} regions and Russia, {len(prices)} rows")


def prepare_municipalities(config):
    folder = config["paths"]["raw"] / "borders"
    table = pd.read_excel(folder / "t_dict_municipal_districts.xlsx", dtype={"oktmo": str})
    table = table[list(MUNICIPALITY_COLUMNS)].rename(columns=MUNICIPALITY_COLUMNS)
    shapes = pyogrio.read_dataframe(folder / "t_dict_municipal_districts_poly.gpkg")
    shapes["territory_id"] = shapes["territory_id"].astype(int)
    shapes = shapes.sort_values("year_from").drop_duplicates("territory_id", keep="last").to_crs(AREA_CRS)
    area = pd.Series(shapes.geometry.area.to_numpy() / 1e6, index=shapes["territory_id"].to_numpy())
    table["area_km2"] = table["territory_id"].map(area).round(3)
    centers = pd.Series(shapes.geometry.representative_point().to_crs("EPSG:4326").to_numpy(), index=shapes["territory_id"].to_numpy())
    missing = table["lat"].isna()
    table.loc[missing, "lat"] = table.loc[missing, "territory_id"].map(lambda t: centers[t].y).round(6)
    table.loc[missing, "lon"] = table.loc[missing, "territory_id"].map(lambda t: centers[t].x).round(6)
    table.to_csv(config["paths"]["prepared"] / "municipalities.csv", index=False)
    outlines = shapes[["territory_id", "geometry"]].copy()
    outlines["geometry"] = outlines.geometry.simplify(1500)
    outlines.to_crs("EPSG:4326").to_parquet(config["paths"]["prepared"] / "shapes.parquet", index=False)
    print(f"municipalities: {table['territory_id'].nunique()} territories, {len(table)} versions, {len(outlines)} outlines")


def prepare(config):
    config["paths"]["prepared"].mkdir(parents=True, exist_ok=True)
    prepare_municipalities(config)
    prepare_prices(config)
    prepare_rosstat(config)