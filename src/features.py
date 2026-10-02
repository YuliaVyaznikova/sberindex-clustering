import io
import zipfile

import numpy as np
import pandas as pd

from .download import FILES

TOTAL = "Все категории"
CATEGORIES = {
    "Продовольствие": "food",
    "Маркетплейсы": "marketplaces",
    "Транспорт": "transport",
    "Здоровье": "health",
    "Общественное питание": "cafe",
}
SECTORS = {
    "А": "agri", "A": "agri",
    "В": "mining", "B": "mining",
    "С": "manufacturing", "C": "manufacturing",
    "D": "utilities", "Е": "utilities", "E": "utilities",
    "F": "construction",
    "G": "trade",
    "Н": "transport", "H": "transport",
    "I": "services", "J": "services", "K": "services", "L": "services", "M": "services", "N": "services", "R": "services", "S": "services",
    "О": "public", "O": "public", "Р": "public", "P": "public", "Q": "public",
}
QUARTERS = {"Январь-март": 1, "Январь-июнь": 2, "Январь-сентябрь": 3, "Январь-декабрь": 4}
AGE_GROUPS = {"Всего": "all", "Моложе трудоспособного возраста": "young", "Старше трудоспособного возраста": "old"}
RUSSIA = "Российская Федерация"
CITY_DISTRICT = "внутригородская территория города федерального значения"
CAPITAL = "административный_центр_субъекта"
LABOR = ["emp_total", "wage", "emp_agri", "emp_mining", "emp_manufacturing", "emp_utilities", "emp_construction", "emp_trade", "emp_transport", "emp_services", "emp_public"]
DEMOGRAPHY = ["population", "urban_share", "share_young", "share_old", "migration"]


def read_package(config, name):
    with zipfile.ZipFile(config["paths"]["raw"] / FILES["package"]) as archive:
        return pd.read_parquet(io.BytesIO(archive.read(f"hackathonlicence/{name}.parquet")))


def read_prepared(config, name):
    return pd.read_csv(config["paths"]["prepared"] / name, dtype={"oktmo": str, "oktmo_stable": str})


def load_spending(config):
    raw = read_package(config, "consumption")
    wide = raw.pivot_table(index=["territory_id", "date"], columns="category", values="value", aggfunc="first")
    wide = wide.rename(columns={**CATEGORIES, TOTAL: "total"}).reset_index()
    wide.columns.name = None
    return wide


def load_territories(config, ids):
    table = read_prepared(config, "municipalities.csv")
    table = table[table["territory_id"].isin(ids) & (table["year_to"] >= 2023)].copy()
    table["oktmo8"] = table["oktmo"].str.replace("-", "").str[:8]
    current = table.sort_values("year_from").drop_duplicates("territory_id", keep="last").set_index("territory_id")
    territories = current[["name", "type", "region", "region_code", "lat", "lon", "area_km2"]].copy()
    territories["capital"] = current["status"].eq(CAPITAL)
    territories["city_district"] = current["type"].eq(CITY_DISTRICT)
    return territories.sort_index(), table[["territory_id", "oktmo8"]]


def price_ratio(config, territories):
    prices = read_prepared(config, "prices.csv")
    prices["date"] = prices["year"].astype(str) + "-" + prices["month"].astype(str).str.zfill(2)
    prices = prices[prices["region"] != RUSSIA].copy()
    prices["price_ratio"] = prices["value"] / prices.groupby("date")["value"].transform("median")
    regions = territories["region"].rename("region").reset_index()
    return regions.merge(prices[["region", "date", "price_ratio"]], on="region")[["territory_id", "date", "price_ratio"]]


def spending_features(monthly, prices):
    table = monthly.merge(prices, on=["territory_id", "date"], how="left")
    table = table.dropna(subset=["total", *CATEGORIES.values()])
    for name in CATEGORIES.values():
        table[f"share_{name}"] = table[name] / table["total"]
    table["share_other"] = 1 - table[[f"share_{name}" for name in CATEGORIES.values()]].sum(axis=1)
    table["spend_rel"] = np.log(table["total"] / table.groupby("date")["total"].transform("median"))
    table["spend_real"] = table["spend_rel"] - np.log(table["price_ratio"])
    table["period"] = pd.PeriodIndex(table["date"], freq="M").asfreq("Q").astype(str)
    columns = [c for c in table.columns if c.startswith("share_")] + ["spend_rel", "spend_real", "price_ratio"]
    grouped = table.groupby(["territory_id", "period"])
    result = grouped[columns].mean()
    result["months"] = grouped.size()
    return result.reset_index()


def match(table, oktmo, keys):
    lookup = oktmo.drop_duplicates("oktmo8").set_index("oktmo8")["territory_id"]
    direct = table["oktmo"].map(lookup)
    table = table.assign(territory_id=direct.fillna(table["oktmo_stable"].map(lookup)), direct=direct.notna())
    table = table.dropna(subset=["territory_id"]).astype({"territory_id": int})
    table = table.sort_values("direct", ascending=False).drop_duplicates(["territory_id", "year", *keys])
    return table.drop(columns="direct")


def labor_features(config, oktmo):
    frames = []
    for name in ["employment", "wages"]:
        table = match(read_prepared(config, f"{name}.csv.gz"), oktmo, ["period", "okved2"])
        frames.append(table.assign(metric=name))
    table = pd.concat(frames)
    table["period"] = table["year"].astype(str) + "Q" + table["period"].map(QUARTERS).astype(str)
    is_total = table["okved2"].str.startswith("Всего")
    totals = table[is_total].pivot_table(index=["territory_id", "period"], columns="metric", values="value", aggfunc="first")
    totals = totals.rename(columns={"employment": "emp_total", "wages": "wage"})
    sectors = table[~is_total & (table["metric"] == "employment")].copy()
    sectors["sector"] = sectors["okved2"].str.split().str[1].map(SECTORS)
    sectors = sectors.pivot_table(index=["territory_id", "period"], columns="sector", values="value", aggfunc="sum")
    shares = sectors.div(totals["emp_total"], axis=0).reindex(totals.index).fillna(0)
    shares.loc[totals["emp_total"].isna()] = np.nan
    shares.columns = [f"emp_{name}" for name in shares.columns]
    return totals.join(shares).reset_index()


def demography_features(config, oktmo):
    population = match(read_prepared(config, "population.csv.gz"), oktmo, ["mest"])
    population = population.pivot_table(index=["territory_id", "year"], columns="mest", values="value", aggfunc="first")
    total = population["Все население"].where(population["Все население"] > 0)
    result = pd.DataFrame({"population": total})
    result["urban_share"] = (population["Городское население"].fillna(0) / total).clip(0, 1)

    age = match(read_prepared(config, "age.csv.gz"), oktmo, ["vozr"])
    age = age.pivot_table(index=["territory_id", "year"], columns="vozr", values="value", aggfunc="first").rename(columns=AGE_GROUPS)
    result["share_young"] = age["young"] / age["all"]
    result["share_old"] = age["old"] / age["all"]
    broken = (result["share_young"] + result["share_old"]) >= 1
    result.loc[broken, ["share_young", "share_old"]] = np.nan

    migration = match(read_prepared(config, "migration.csv.gz"), oktmo, [])
    migration["year"] = migration["year"] + 1
    result = result.join(migration.set_index(["territory_id", "year"])["value"].rename("migration"))
    result["migration"] = 1000 * result["migration"] / result["population"]
    result = result.reset_index()
    result["year"] = result["year"].astype(str)
    return result


def market_access(config, territories):
    values = read_package(config, "market_access").set_index("territory_id")["market_access"].reindex(territories.index)
    known = territories.loc[values.notna(), ["lat", "lon"]]
    for tid in values.index[values.isna()]:
        lat, lon = territories.at[tid, "lat"], territories.at[tid, "lon"]
        distance = (known["lat"] - lat) ** 2 + ((known["lon"] - lon) * np.cos(np.radians(lat))) ** 2
        values[tid] = values[distance.idxmin()]
    return values


def fill_from_neighbours(table, columns):
    table = table.sort_values(["territory_id", "period"])
    table[columns] = table.groupby("territory_id")[columns].transform(lambda s: s.ffill().bfill())
    return table


def fill_from_peers(table, columns, territories):
    region = table["territory_id"].map(territories["region"])
    peers = region + "|" + table["territory_id"].map(territories["type"])
    for column in columns:
        table[column] = table[column].fillna(table.groupby(peers)[column].transform("median"))
        table[column] = table[column].fillna(table.groupby(region)[column].transform("median"))
        table[column] = table[column].fillna(table[column].median())
    return table


def month_number(date):
    year, month = map(int, date.split("-"))
    return 12 * year + month - 1


def month_label(number):
    return f"{number // 12}-{number % 12 + 1:02d}"


def windows(monthly):
    dates = sorted(monthly["date"].unique())
    first = month_number(dates[0]) + 11
    ends = [d for d in dates if month_number(d) >= first and int(d[5:]) % 3 == 0]
    return {f"{d[:4]}Q{int(d[5:]) // 3}": d for d in ends}


def rolling_spending(monthly, prices, periods):
    table = monthly.merge(prices, on=["territory_id", "date"], how="left").dropna(subset=["total", *CATEGORIES.values()])
    frames = []
    for period, end in periods.items():
        last = month_number(end)
        months = [month_label(m) for m in range(last - 11, last + 1)]
        grouped = table[table["date"].isin(months)].groupby("territory_id")
        sums = grouped[["total", *CATEGORIES.values()]].sum()
        result = pd.DataFrame({f"share_{name}": sums[name] / sums["total"] for name in CATEGORIES.values()})
        result["share_other"] = 1 - result.sum(axis=1)
        level = grouped["total"].mean()
        result["spend_rel"] = np.log(level / level.median())
        result["price_ratio"] = grouped["price_ratio"].mean()
        result["spend_real"] = result["spend_rel"] - np.log(result["price_ratio"])
        result["months"] = grouped.size()
        frames.append(result.reset_index().assign(period=period))
    return pd.concat(frames, ignore_index=True)


def trailing_year(values, year, quarter):
    current = values.get((year, quarter))
    if quarter == 4:
        return current
    previous_year, previous_part = values.get((year - 1, 4)), values.get((year - 1, quarter))
    if current is None or previous_year is None or previous_part is None:
        return current
    return (3 * quarter * current + 12 * previous_year - 3 * quarter * previous_part) / 12


def rolling_labor(config, oktmo, periods):
    employment = match(read_prepared(config, "employment.csv.gz"), oktmo, ["period", "okved2"])
    wages = match(read_prepared(config, "wages.csv.gz"), oktmo, ["period", "okved2"])
    employment["sector"] = employment["okved2"].str.split().str[1].map(SECTORS)
    employment.loc[employment["okved2"].str.startswith("Всего"), "sector"] = "total"
    employment["quarter"] = employment["period"].map(QUARTERS)
    wages = wages[wages["okved2"].str.startswith("Всего")].assign(quarter=lambda t: t["period"].map(QUARTERS))
    counts = employment.dropna(subset=["sector"]).groupby(["territory_id", "sector", "year", "quarter"])["value"].sum()
    wage = wages.set_index(["territory_id", "year", "quarter"])["value"]
    sectors = list(dict.fromkeys(SECTORS.values()))
    rows = []
    for tid, part in counts.groupby(level="territory_id"):
        series = {sector: group.droplevel(["territory_id", "sector"]).to_dict() for sector, group in part.groupby(level="sector")}
        total = series.get("total", {})
        payroll = {key: wage[(tid, *key)] * value for key, value in total.items() if (tid, *key) in wage.index}
        for period in periods:
            year, quarter = int(period[:4]), int(period[-1])
            people = trailing_year(total, year, quarter)
            if not people or people <= 0:
                continue
            row = {"territory_id": tid, "period": period, "emp_total": people}
            for sector in sectors:
                value = trailing_year(series.get(sector, {}), year, quarter)
                row[f"emp_{sector}"] = max(value, 0) / people if value is not None else 0.0
            pay = trailing_year(payroll, year, quarter)
            row["wage"] = pay / people if pay is not None and pay > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def rolling_demography(config, oktmo, periods):
    yearly = demography_features(config, oktmo).set_index(["territory_id", "year"])
    columns = ["population", "urban_share", "share_young", "share_old"]
    years = sorted(yearly.index.get_level_values("year").unique())
    first, second = yearly.xs(years[0], level="year")[columns], yearly.xs(years[-1], level="year")[columns]
    index = first.index.union(second.index)
    first, second = first.reindex(index), second.reindex(index)
    first, second = first.fillna(second), second.fillna(first)
    migration = yearly["migration"].groupby(level="territory_id").mean().reindex(index)
    start = month_number(f"{years[0]}-01")
    frames = []
    for period, end in periods.items():
        weight = float(np.clip((month_number(end) - 5.5 - start) / 12, 0, 1))
        frame = (1 - weight) * first + weight * second
        frame["migration"] = migration
        frames.append(frame.reset_index().assign(period=period))
    return pd.concat(frames, ignore_index=True)


def finish(table, territories, config):
    table["market_access"] = table["territory_id"].map(market_access(config, territories))
    table = fill_from_neighbours(table, LABOR + DEMOGRAPHY)
    table["wage_rel"] = np.log(table["wage"] / table.groupby("period")["wage"].transform("median"))
    table["wage_real"] = table["wage_rel"] - np.log(table["price_ratio"])
    table["emp_rate"] = table["emp_total"] / table["population"]
    table["log_population"] = np.log(table["population"])
    table["log_density"] = np.log(table["population"] / table["territory_id"].map(territories["area_km2"]))
    derived = ["wage_rel", "wage_real", "emp_rate", "log_population", "log_density", "market_access"]
    table = fill_from_peers(table, LABOR + DEMOGRAPHY + derived, territories)
    return table.sort_values(["territory_id", "period"]).reset_index(drop=True)


def quarterly_table(config, monthly, territories, oktmo, prices):
    table = spending_features(monthly, prices)
    table = table.merge(labor_features(config, oktmo), on=["territory_id", "period"], how="left")
    table["year"] = table["period"].str[:4]
    table = table.merge(demography_features(config, oktmo), on=["territory_id", "year"], how="left").drop(columns="year")
    table["complete"] = table["months"] == 3
    return finish(table, territories, config)


def rolling_table(config, monthly, territories, oktmo, prices):
    periods = windows(monthly)
    table = rolling_spending(monthly, prices, periods)
    table = table.merge(rolling_labor(config, oktmo, periods), on=["territory_id", "period"], how="left")
    table = table.merge(rolling_demography(config, oktmo, periods), on=["territory_id", "period"], how="left")
    table["complete"] = table["months"] == 12
    return finish(table, territories, config)


def build_features(config):
    out = config["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    monthly = load_spending(config)
    territories, oktmo = load_territories(config, monthly["territory_id"].unique())
    prices = price_ratio(config, territories)
    quarterly = quarterly_table(config, monthly, territories, oktmo, prices)
    rolling = rolling_table(config, monthly, territories, oktmo, prices)
    territories["full_series"] = monthly.dropna().groupby("territory_id").size().reindex(territories.index).eq(24)
    quarterly.to_parquet(out / "features.parquet", index=False)
    rolling.to_parquet(out / "features_rolling.parquet", index=False)
    territories.to_csv(out / "territories.csv")
    monthly.to_parquet(out / "spending_monthly.parquet", index=False)
    for name, table in [("quarters", quarterly), ("rolling years", rolling)]:
        complete = table[table["complete"]].groupby("period").size()
        print(f"features by {name}: {table['territory_id'].nunique()} municipalities, complete per period {complete.to_dict()}")