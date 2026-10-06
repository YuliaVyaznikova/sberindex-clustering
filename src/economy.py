import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from .features import CATEGORIES
from .network import BLOCKS, SPEND_SHARES, transform

ECONOMY = BLOCKS["labor"] + BLOCKS["place"]
NAMES = list(CATEGORIES.values())
GAP_STEPS = [
    (-9, -0.15, "#184f95", "ниже на 15% и больше"),
    (-0.15, -0.05, "#6da7ec", "ниже на 5-15%"),
    (-0.05, 0.05, "#b9b8b2", "в пределах 5%"),
    (0.05, 0.15, "#ec8a85", "выше на 5-15%"),
    (0.15, 9, "#a02d2d", "выше на 15% и больше"),
]
SCHEMES = {
    "monthly_independent": ("типы по каждому месяцу отдельно", "#eb6834", "-"),
    "monthly_pooled": ("месяц, общие центры", "#eb6834", "--"),
    "rolling_independent": ("скользящий год, каждое окно отдельно", "#2a78d6", "-"),
    "rolling_pooled": ("скользящий год, общие центры", "#2a78d6", "--"),
}


def spending_gap(features, territories, types, period, seed=0):
    rows = features["period"] == period
    scaled = transform(features).loc[rows].set_axis(features.loc[rows, "territory_id"].to_numpy())
    level = features.loc[rows].set_index("territory_id")["spend_real"]
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed)
    predicted = cross_val_predict(model, scaled[ECONOMY], level, cv=KFold(5, shuffle=True, random_state=seed))
    assigned = types[types["period"] == period].set_index("territory_id")["type"]
    table = pd.DataFrame({
        "name": territories.loc[level.index, "name"], "region": territories.loc[level.index, "region"],
        "type": assigned.reindex(level.index), "gap": np.exp(level.to_numpy() - predicted) - 1,
    }, index=level.index)
    return table, r2_score(level, predicted)


def gap_stability(table, r2, early):
    common = table.index.intersection(early.index)
    return pd.DataFrame([{
        "r2_out_of_fold": r2, "above_15": int((table["gap"] >= 0.15).sum()), "below_15": int((table["gap"] <= -0.15).sum()),
        "spearman_first_last": spearmanr(early.loc[common, "gap"], table.loc[common, "gap"]).statistic, "municipalities": len(common),
    }])


def short_name(name):
    for word in ["муниципальный район", "муниципальный округ", "городской округ город", "городской округ"]:
        name = name.replace(word, "")
    return " ".join(name.split())


def plot_gap(shapes, table, r2, period, colors, path):
    shapes = shapes.assign(gap=shapes["node"].map(table["gap"]))
    fig = plt.figure(figsize=(16, 7.4))
    ax = fig.add_axes((0.0, 0.06, 0.72, 0.84))
    shapes.plot(ax=ax, color=colors["background"], edgecolor=colors["edge"], linewidth=0.1)
    handles = []
    for low, high, color, label in GAP_STEPS:
        shapes[(shapes["gap"] >= low) & (shapes["gap"] < high)].plot(ax=ax, color=color, edgecolor=colors["surface"], linewidth=0.1)
        count = int(table["gap"].between(low, high, inclusive="left").sum())
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f"{label}, {count} МО"))
    ax.legend(handles=handles, loc="lower left", frameon=False, title="траты жителей против прогноза\nпо местной экономике", fontsize=9, title_fontsize=9.5)
    ax.set_axis_off()
    side = fig.add_axes((0.73, 0.06, 0.26, 0.84))
    side.set_axis_off()
    y = 1.0
    for title, rows, color in [("Тратят больше прогноза", table.nlargest(8, "gap"), "#a02d2d"), ("Тратят меньше прогноза", table.nsmallest(8, "gap"), "#184f95")]:
        side.text(0, y, title, fontsize=10.5, weight="bold", va="top", transform=side.transAxes)
        y -= 0.055
        for _, row in rows.iterrows():
            side.text(0, y, f"{row['gap']:+.0%}", color=color, fontsize=9.5, va="top", transform=side.transAxes)
            side.text(0.13, y, f"{short_name(row['name'])}, {row['region']}", fontsize=9, va="top", transform=side.transAxes)
            y -= 0.045
        y -= 0.04
    fig.suptitle(f"Где траты не следуют за местной экономикой, окно {period}", x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.015, f"Прогноз уровня трат по {len(ECONOMY)} признакам труда и места, градиентный бустинг, out-of-fold на 5-fold кросс-валидации, R² = {r2:.2f}. "
             "Светло-серым цветом показаны МО без полных данных.", color=colors["muted"], fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def spending_features(sums, months):
    shares = pd.DataFrame({f"share_{n}": sums[n] / sums["total"] for n in NAMES})
    shares["share_other"] = 1 - shares.sum(axis=1)
    logs = np.log(shares[SPEND_SHARES].clip(lower=1e-4))
    result = logs.sub(logs.mean(axis=1), axis=0)
    level = sums["total"] / months
    result["level"] = np.log(level / level.median())
    return result


def pooled_scale(frames):
    stacked = pd.concat(frames.values())
    low, high = stacked.quantile(0.01), stacked.quantile(0.99)
    stacked = stacked.clip(low, high, axis=1)
    mean, std = stacked.mean(), stacked.std()
    return {m: ((f.clip(low, high, axis=1) - mean) / std).to_numpy() for m, f in frames.items()}


def align(reference, labels, k):
    overlap = np.zeros((k, k))
    np.add.at(overlap, (reference, labels), 1)
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.empty(k, dtype=int)
    mapping[cols] = rows
    return mapping[labels]


def independent(scaled, k):
    labels, previous = {}, None
    for month, x in scaled.items():
        current = KMeans(k, n_init=10, random_state=0).fit_predict(x)
        labels[month] = current if previous is None else align(previous, current, k)
        previous = labels[month]
    return labels


def pooled(scaled, k):
    model = KMeans(k, n_init=10, random_state=0).fit(np.vstack(list(scaled.values())))
    return {m: model.predict(x) for m, x in scaled.items()}


def change_shares(labels):
    months = list(labels)
    return pd.Series({months[t]: float(np.mean(labels[months[t - 1]] != labels[months[t]])) for t in range(1, len(months))})


def flip_back(labels, horizon=2):
    months = list(labels)
    changed = returned = 0
    for t in range(1, len(months) - horizon):
        before, moved = labels[months[t - 1]], labels[months[t - 1]] != labels[months[t]]
        back = np.zeros_like(moved)
        for h in range(1, horizon + 1):
            back |= labels[months[t + h]] == before
        changed += moved.sum()
        returned += (moved & back).sum()
    return returned / changed


def monthly_transitions(monthly, territories, k, window=12):
    monthly = monthly.dropna(subset=["total", *NAMES])
    months = sorted(monthly["date"].unique())
    full = monthly.groupby("territory_id")["date"].nunique()
    ids = full.index[(full == len(months)) & ~full.index.map(territories["city_district"]).astype(bool)]
    by_month = {m: spending_features(monthly[monthly["date"] == m].set_index("territory_id").loc[ids], 1) for m in months}
    by_year = {}
    for end in range(window - 1, len(months)):
        chosen = months[end - window + 1:end + 1]
        sums = monthly[monthly["date"].isin(chosen)].groupby("territory_id")[["total", *NAMES]].sum().loc[ids]
        by_year[months[end]] = spending_features(sums, window)
    by_month, by_year = pooled_scale(by_month), pooled_scale(by_year)
    schemes = {
        "monthly_independent": independent(by_month, k), "monthly_pooled": pooled(by_month, k),
        "rolling_independent": independent(by_year, k), "rolling_pooled": pooled(by_year, k),
    }
    shares = pd.DataFrame({name: change_shares(labels) for name, labels in schemes.items()}).rename_axis("month")
    summary = []
    for name, labels in schemes.items():
        stacked = np.stack(list(labels.values()))
        summary.append({"scheme": name, "mean_change": shares[name].mean(), "flip_back_2m": flip_back(labels),
                        "ever_changed": float(np.mean((stacked != stacked[0]).any(axis=0))), "changed_first_last": float(np.mean(stacked[0] != stacked[-1]))})
    deviations = np.stack([by_month[m] for m in months])
    deviations = deviations - deviations.mean(axis=0, keepdims=True)
    common = np.median(deviations, axis=1, keepdims=True)
    seasonal = 1 - ((deviations - common) ** 2).sum() / (deviations ** 2).sum()
    first, second = deviations[:12], deviations[12:24]
    pairs = [(first[:, i, j], second[:, i, j]) for i in range(first.shape[1]) for j in range(first.shape[2])]
    repeat = np.nanmedian([np.corrcoef(a, b)[0, 1] for a, b in pairs if a.std() > 0 and b.std() > 0])
    return shares, pd.DataFrame(summary), {"municipalities": len(ids), "common_seasonal_share": seasonal, "year_to_year_pattern_correlation": repeat}


def plot_monthly(shares, k, municipalities, colors, path):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for name, (label, color, line) in SCHEMES.items():
        series = shares[name].dropna()
        ax.plot([list(shares.index).index(m) for m in series.index], series.to_numpy(), line, color=color, marker="o", markersize=3.5, label=label)
    ticks = list(shares.index)
    ax.set_xticks(list(range(len(ticks)))[::2], ticks[::2], fontsize=8.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("доля МО, сменивших тип за месяц")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Помесячные «переходы» между типами в основном сезонность и шум", loc="left")
    fig.text(0.01, 0.01, f"{municipalities} МО с тратами за все месяцы, k-means, k = {k}, только признаки трат. Подпись месяца означает конец перехода.", color=colors["muted"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def marketplaces(monthly, types, period):
    assigned = types[types["period"] == period].set_index("territory_id")["type"]
    monthly = monthly.dropna(subset=["total", "marketplaces"])
    monthly = monthly[monthly["territory_id"].isin(assigned.index)]
    full = monthly.groupby("territory_id")["date"].nunique()
    monthly = monthly[monthly["territory_id"].isin(full.index[full == full.max()])]
    monthly = monthly.assign(share=monthly["marketplaces"] / monthly["total"], type=monthly["territory_id"].map(assigned))
    shares = monthly.pivot_table(index="date", columns="type", values="share", aggfunc="median")
    rubles = monthly.pivot_table(index="date", columns="type", values="marketplaces", aggfunc="median")
    return shares, rubles, monthly["territory_id"].nunique()


def catch_up(shares, rubles, cities):
    first, last = shares.iloc[:12].mean(), shares.iloc[-12:].mean()
    return pd.DataFrame({
        "share_first_year": first, "share_last_year": last, "gain_pp": (last - first) * 100,
        "ratio_to_cities_first_year": first / first[cities], "ratio_to_cities_last_year": last / last[cities],
        "rubles_growth": rubles.iloc[-12:].mean() / rubles.iloc[:12].mean(),
    })


def plot_marketplaces(shares, palette, municipalities, period, colors, path):
    order, cities = palette["order"], palette["order"][0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for t in order:
        ax.plot(range(len(shares)), shares[t].to_numpy(), color=palette["colors"][t], marker="o", markersize=3, label=palette["short"][t])
    ax.set_xticks(range(0, len(shares), 3), list(shares.index)[::3], fontsize=8.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("медианная доля маркетплейсов в тратах")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.set_title("Доля маркетплейсов растёт во всех типах", loc="left")
    ax = axes[1]
    ratio = shares.div(shares[cities], axis=0)
    for t in order[1:]:
        ax.plot(range(len(ratio)), ratio[t].to_numpy(), color=palette["colors"][t], marker="o", markersize=3, label=palette["short"][t])
    ax.axhline(1, color=colors["muted"], linewidth=0.8)
    ax.set_xticks(range(0, len(ratio), 6), list(ratio.index)[::6], fontsize=8.5)
    ax.set_ylabel(f"доля маркетплейсов к доле в типе «{palette['short'][cities]}»")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Периферия впереди Городов, удалённые догоняют", loc="left")
    fig.text(0.01, 0.01, f"{municipalities} МО с тратами за все месяцы, типы по окну {period}, медиана по МО типа.", color=colors["muted"], fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)