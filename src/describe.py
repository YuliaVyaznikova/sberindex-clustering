import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path as CurvePath
from scipy.stats import kruskal

from .download import FILES
from .economy import catch_up, gap_stability, marketplaces, monthly_transitions, plot_gap, plot_marketplaces, plot_monthly, spending_gap
from .network import BLOCKS, CITIES, EARTH_RADIUS_KM, load_data, transform

SURFACE, INK, MUTED, GRID, BACKGROUND = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#ebeae5"
MAP_CRS = "+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=100 +datum=WGS84 +units=m"
SUMMARY = {
    "share_food": "еда", "share_marketplaces": "маркетплейсы", "share_cafe": "общепит", "share_transport": "транспорт",
    "share_health": "здоровье", "spend_real": "траты", "wage_real": "зарплата", "emp_rate": "занятые на жителя",
    "emp_agri": "агро", "emp_mining": "добыча", "emp_manufacturing": "обработка", "emp_public": "бюджет",
    "emp_services": "услуги", "emp_trade": "торговля", "urban_share": "горожане", "log_density": "плотность",
    "share_young": "моложе трудоспособного", "share_old": "старше трудоспособного", "migration": "миграция", "market_access": "доступность рынков",
}
PROFILE = {
    "share_food": "доля еды", "share_marketplaces": "доля маркетплейсов", "share_cafe": "доля общепита", "share_transport": "доля транспорта",
    "share_health": "доля здоровья", "spend_real": "уровень трат", "emp_agri": "занятость, агро", "emp_mining": "занятость, добыча",
    "emp_manufacturing": "занятость, обработка", "emp_public": "занятость, бюджет", "emp_services": "занятость, услуги",
    "wage_real": "зарплата", "emp_rate": "занятые на жителя", "log_population": "население", "urban_share": "доля горожан",
    "log_density": "плотность", "share_young": "доля молодых", "share_old": "доля пожилых", "migration": "миграция", "market_access": "доступность рынков",
}
RULE_NAMES = {
    "share_food": "доля еды в тратах", "share_marketplaces": "доля маркетплейсов", "share_transport": "доля транспорта",
    "share_health": "доля здоровья", "share_cafe": "доля общепита", "share_other": "доля прочего",
    "spend_real": "траты на жителя к медиане МО", "wage_real": "зарплата к медиане МО",
    "emp_rate": "работников организаций на жителя", "emp_agri": "занятость в сельском хозяйстве", "emp_mining": "занятость в добыче",
    "emp_manufacturing": "занятость в обработке", "emp_utilities": "занятость в энергетике и ЖКХ", "emp_construction": "занятость в стройке",
    "emp_trade": "занятость в торговле", "emp_transport": "занятость в транспорте", "emp_services": "занятость в услугах",
    "emp_public": "занятость в бюджетном секторе", "log_population": "население", "urban_share": "доля горожан",
    "log_density": "плотность", "share_young": "доля моложе трудоспособного", "share_old": "доля старше трудоспособного",
    "migration": "миграционный прирост на 1000 жителей", "market_access": "индекс доступности рынков",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def window_rows(features, types, period):
    assigned = types[types["period"] == period].set_index("territory_id")["type"]
    rows = features[features["period"] == period].set_index("territory_id").loc[assigned.index]
    return rows, assigned


def summarize(features, types, period):
    rows, assigned = window_rows(features, types, period)
    table = rows.groupby(assigned)[list(SUMMARY)].median().rename(columns=SUMMARY)
    for column in ["траты", "зарплата", "плотность"]:
        table[column] = np.exp(table[column])
    population = rows["population"].groupby(assigned).sum()
    table.insert(0, "МО", assigned.value_counts())
    table.insert(1, "население, млн", population / 1e6)
    table.insert(2, "доля населения", population / population.sum())
    return table


def profiles(features, types, period):
    scaled = transform(features)
    scaled["territory_id"], scaled["period"] = features["territory_id"].to_numpy(), features["period"].to_numpy()
    rows = scaled[scaled["period"] == period].set_index("territory_id")
    assigned = types[types["period"] == period].set_index("territory_id")["type"]
    return rows.loc[assigned.index, list(PROFILE)].groupby(assigned).mean()


def examples(features, territories, types, period, top=6):
    rows, assigned = window_rows(features, types, period)
    result = {}
    for t, members in assigned.groupby(assigned):
        biggest = rows.loc[members.index, "population"].sort_values(ascending=False).head(top).index
        regions = territories.loc[members.index, "region"].value_counts().head(4)
        result[t] = {
            "крупнейшие МО": "; ".join(territories.loc[biggest, "name"]),
            "чаще всего в регионах": ", ".join(f"{r} {n}" for r, n in regions.items()),
        }
    return pd.DataFrame(result).T


def changes(features, types, first, last):
    a = features[features["period"] == first].set_index("territory_id")
    b = features[features["period"] == last].set_index("territory_id")
    assigned = types[types["period"] == last].set_index("territory_id")["type"]
    common = assigned.index.intersection(a.index).intersection(b.index)
    shares = {"еда": "share_food", "маркетплейсы": "share_marketplaces", "транспорт": "share_transport", "здоровье": "share_health", "общепит": "share_cafe", "прочее": "share_other"}
    delta = pd.DataFrame({f"{name}, п.п.": 100 * (b.loc[common, column] - a.loc[common, column]) for name, column in shares.items()})
    delta["уровень трат, %"] = 100 * (np.exp(b.loc[common, "spend_real"] - a.loc[common, "spend_real"]) - 1)
    delta["зарплата, %"] = 100 * (np.exp(b.loc[common, "wage_real"] - a.loc[common, "wage_real"]) - 1)
    return delta.groupby(assigned.loc[common]).median()


def load_mobility(config, territories):
    raw = pd.read_csv(config["paths"]["raw"] / FILES["mobility"])
    raw = raw[raw["period"].str.startswith(config["describe"]["mobility_year"])]
    local =territories[territories["region"].isin(config["describe"]["mobility_regions"])]
    names = local.rename_axis("territory_id").reset_index().groupby("name")["territory_id"].unique()
    unique = names[names.map(len) == 1].map(lambda ids: ids[0])
    raw = raw.assign(territory_id=raw["ref_area"].map(unique)).dropna(subset=["territory_id"])
    return raw.set_index(raw["territory_id"].astype(int))["value"]


def mobility_check(mobility, types, period, minimum=5):
    assigned = types[types["period"] == period].set_index("territory_id")["type"]
    joined = pd.DataFrame({"type": assigned, "km": mobility}).dropna()
    shown = joined.groupby("type").filter(lambda g: len(g) >= minimum)
    statistic, p_value = kruskal(*[g["km"].to_numpy() for _, g in shown.groupby("type")])
    return shown, statistic, p_value


def municipality_shapes(config, original):
    shapes = geopandas.read_parquet(config["paths"]["prepared"] / "shapes.parquet").to_crs(MAP_CRS)
    city_of = {d: CITIES[original.at[d, "region"]] for d in original.index[original["city_district"]]}
    shapes["node"] = shapes["territory_id"].map(lambda t: city_of.get(t, t))
    return shapes


def plot_map(shapes, summary, assigned, period, palette, path):
    shapes = shapes.assign(type=shapes["node"].map(assigned))
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.6))
    for ax, t in zip(axes.ravel(), palette["order"]):
        shapes.plot(ax=ax, color=BACKGROUND, edgecolor=SURFACE, linewidth=0.1)
        shapes[shapes["type"] == t].plot(ax=ax, color=palette["colors"][t], edgecolor=SURFACE, linewidth=0.1)
        ax.set_axis_off()
        ax.set_title(f"{palette['names'][t]}\n{int(summary.at[t, 'МО'])} МО, {summary.at[t, 'доля населения']:.0%} населения", loc="left", fontsize=10.5)
    fig.suptitle(f"Типы местных экономик, скользящий год с концом в {period}", x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.01, "Серым цветом показаны остальные типы и МО без полных данных. Районы Москвы и Санкт-Петербурга склеены в два узла.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ribbon(ax, x0, y0, x1, y1, height, color):
    middle = (x0 + x1) / 2
    vertices = [(x0, y0), (middle, y0), (middle, y1), (x1, y1), (x1, y1 + height), (middle, y1 + height), (middle, y0 + height), (x0, y0 + height), (x0, y0)]
    codes = [CurvePath.MOVETO] + [CurvePath.CURVE4] * 3 + [CurvePath.LINETO] + [CurvePath.CURVE4] * 3 + [CurvePath.CLOSEPOLY]
    ax.add_patch(PathPatch(CurvePath(vertices, codes), facecolor=color, edgecolor="none", alpha=0.45))


def plot_alluvial(wide, palette, path, gap=40, width=0.12):
    periods, order, colors = list(wide.columns), palette["order"], palette["colors"]
    fig, ax = plt.subplots(figsize=(14, 7.5))
    tops = {}
    for x, period in enumerate(periods):
        y, counts = 0, wide[period].value_counts()
        for t in order:
            n = int(counts.get(t, 0))
            tops[(period, t)] = y
            ax.add_patch(plt.Rectangle((x - width / 2, y), width, n, facecolor=colors[t], edgecolor=SURFACE, linewidth=2))
            if x == 0:
                ax.text(x - width / 2 - 0.04, y + n / 2, f"{palette['names'][t]}  {n}", ha="right", va="center", fontsize=9.5)
            if x == len(periods) - 1:
                ax.text(x + width / 2 + 0.04, y + n / 2, str(n), ha="left", va="center", fontsize=9.5, color=MUTED)
            y += n + gap
        ax.text(x, -60, period, ha="center", va="top", fontsize=9.5, color=MUTED)
    for x, (a, b) in enumerate(zip(periods, periods[1:])):
        out, into = {t: 0 for t in order}, {t: 0 for t in order}
        flows = wide.groupby([a, b]).size()
        for s in order:
            for d in order:
                n = int(flows.get((s, d), 0))
                if n:
                    ribbon(ax, x + width / 2, tops[(a, s)] + out[s], x + 1 - width / 2, tops[(b, d)] + into[d], n, colors[s])
                    out[s] += n
                    into[d] += n
    changed = float((wide[periods[0]] != wide[periods[-1]]).mean())
    ax.set_xlim(-1.9, len(periods) - 0.55)
    ax.set_ylim(tops[(periods[0], order[-1])] + 300, -160)
    ax.set_axis_off()
    ax.set_title(f"Переходы между типами по окнам скользящего года, {len(wide)} МО. С {periods[0]} по {periods[-1]} сменили тип {changed:.1%}", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_transitions(wide, palette, path):
    order = palette["order"]
    first, last = wide.columns[0], wide.columns[-1]
    table = pd.crosstab(wide[first], wide[last]).reindex(index=order, columns=order, fill_value=0).to_numpy()
    moved = table.copy()
    np.fill_diagonal(moved, 0)
    cmap = LinearSegmentedColormap.from_list("sequential", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.imshow(np.ma.masked_where(np.eye(len(order), dtype=bool), moved), cmap=cmap, vmin=0, vmax=max(moved.max(), 1))
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=BACKGROUND, edgecolor=SURFACE, linewidth=2))
                ax.text(j, i, f"остались\n{table[i, i]}", ha="center", va="center", fontsize=8, color=MUTED)
            elif moved[i, j]:
                ax.text(j, i, str(moved[i, j]), ha="center", va="center", fontsize=10, color="white" if moved[i, j] > moved.max() / 2 else INK)
    ax.set_xticks(range(len(order)), [palette["short"][t] for t in order], rotation=25, ha="right")
    ax.set_yticks(range(len(order)), [palette["names"][t] for t in order])
    ax.set_xlabel(f"тип в {last}")
    ax.set_ylabel(f"тип в {first}")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Сменили тип с {first} по {last}, {int(moved.sum())} из {int(table.sum())} МО", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profiles(means, palette, period, path):
    order = palette["order"]
    columns = list(PROFILE)
    values = means.loc[order, columns]
    cmap = LinearSegmentedColormap.from_list("diverging", ["#184f95", "#6da7ec", "#f0efec", "#ec8a85", "#a02d2d"])
    fig, ax = plt.subplots(figsize=(15, 5.4))
    image = ax.imshow(values.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -1.6, 1.6), aspect="auto")
    ax.set_xticks(range(len(columns)), [PROFILE[c] for c in columns], rotation=40, ha="right")
    ax.set_yticks(range(len(order)), [palette["names"][t] for t in order])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values.iat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 1.0 else INK)
    bounds = np.cumsum([sum(c in BLOCKS[b] for c in columns) for b in ["spending", "labor", "place"]])
    for edge in bounds[:-1]:
        ax.axvline(edge - 0.5, color=SURFACE, linewidth=4)
    for start, end, name in zip([0, *bounds[:-1]], bounds, ["потребление", "труд", "место"]):
        ax.text((start + end - 1) / 2, -0.75, name, ha="center", va="bottom", fontsize=9.5, color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    bar.set_label("отклонение от среднего по МО, ст. откл.")
    bar.outline.set_visible(False)
    ax.set_title(f"Профили типов, окно {period}, отличие от среднего МО", loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_changes(delta, palette, first, last, path):
    order = palette["order"][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, column, title, unit in [
        (axes[0], "маркетплейсы, п.п.", f"Рост доли маркетплейсов в тратах, {first} -> {last}", "п.п."),
        (axes[1], "уровень трат, %", "Изменение уровня трат относительно медианного МО", "%"),
    ]:
        values = delta.loc[order, column]
        ax.barh(range(len(order)), values.to_numpy(), color=[palette["colors"][t] for t in order], height=0.6)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        for i, v in enumerate(values.to_numpy()):
            ax.text(v + (0.08 if v >= 0 else -0.08), i, f"{v:+.1f} {unit}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
        ax.set_yticks(range(len(order)), [palette["names"][t] for t in order])
        ax.set_title(title, loc="left")
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        span = max(abs(values.min()), abs(values.max())) * 1.35
        ax.set_xlim(min(0.0, values.min() - span * 0.3), span)
    fig.text(0.01, 0.01, "Медианы по МО каждого типа. Уровень трат с поправкой на цены региона.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mobility(shown, statistic, p_value, palette, path):
    order = [t for t in palette["order"] if t in set(shown["type"])][::-1]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for i, t in enumerate(order):
        values = shown.loc[shown["type"] == t, "km"].to_numpy()
        ax.scatter(values, i + rng.uniform(-0.18, 0.18, len(values)), s=16, color=palette["colors"][t], edgecolor=SURFACE, linewidth=0.6)
        median = np.median(values)
        ax.plot([median, median], [i - 0.3, i + 0.3], color=INK, linewidth=2)
        ax.text(median, i + 0.36, f"медиана {median:.2f} км, {len(values)} МО", fontsize=8.5, color=MUTED, ha="center")
    ax.set_yticks(range(len(order)), [palette["names"][t] for t in order])
    ax.set_xscale("log")
    ax.set_xlabel("индекс покупательской мобильности, км (логарифмическая шкала)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Внешняя проверка, радиус покупок по типам", loc="left")
    fig.text(0.01, 0.01, f"Индекс мобильности не входил в признаки. Краскел-Уоллис, H = {statistic:.1f}, p = {p_value:.1g}. Показаны типы, у которых есть хотя бы 5 МО с индексом.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def approach_label(name):
    family, _, value = name.partition(" ")
    return {
        "independent": "независимо по окнам",
        "evolutionary": f"сглаживание, память {value}",
        "affect": "AFFECT",
        "supra": f"SPECTRA, многослойная, связь {value.replace(' + refinement', '')} и уточнение" if "refinement" in value else f"многослойная, связь {value}",
        "fixed": "фиксированная типология",
    }[family]


def plot_dynamics(approaches, path):
    colors = {"independent": INK, "evolutionary": "#eb6834", "affect": "#eda100", "supra": "#2a78d6", "fixed": "#1baf7a"}
    rows = approaches.iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, column, scale, title, unit in [
        (axes[0], "changed_first_last", 100, "Сменили тип между первым и последним окном", "%"),
        (axes[1], "MQ", 1, "Качество снимков, MQ на графе потребления", ""),
    ]:
        values = rows[column] * scale
        color = [colors[name.split()[0]] for name in rows["approach"]]
        ax.hlines(range(len(rows)), values.min() * (0 if column == "changed_first_last" else 0.98), values, color=GRID, linewidth=2)
        ax.scatter(values, range(len(rows)), s=55, color=color, edgecolor=SURFACE, linewidth=1, zorder=3)
        for i, v in enumerate(values):
            ax.text(v, i + 0.28, f"{v:.0f}{unit}" if unit else f"{v:.3f}", ha="center", fontsize=8, color=MUTED)
        ax.set_title(title, loc="left")
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows)), [approach_label(name) for name in rows["approach"]])
    fig.text(0.01, 0.01, "Одна и та же модель внутри окна, различается связь между окнами. МО, которые есть во всех окнах.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confidence(shapes, confidence, period, path):
    values = confidence[confidence["period"] == period].set_index("territory_id")["confidence"]
    steps = [(0.0, "#86b6ef", "меньше 0.5"), (0.5, "#3987e5", "0.5-0.7"), (0.7, "#1c5cab", "0.7-0.9"), (0.9, "#0d366b", "0.9 и выше")]
    shapes = shapes.assign(confidence=shapes["node"].map(values))
    fig, ax = plt.subplots(figsize=(12, 6.4))
    shapes.plot(ax=ax, color=BACKGROUND, edgecolor=SURFACE, linewidth=0.1)
    handles = []
    for (low, color, label), high in zip(steps, [s[0] for s in steps[1:]] + [1.01]):
        part = shapes[(shapes["confidence"] >= low) & (shapes["confidence"] < high)]
        part.plot(ax=ax, color=color, edgecolor=SURFACE, linewidth=0.1)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f"{label}, {int(((values >= low) & (values < high)).sum())} МО"))
    ax.legend(handles=handles, loc="lower left", frameon=False, title="доля прогонов в своём типе", fontsize=9)
    ax.set_axis_off()
    ax.set_title(f"Уверенность принадлежности к типу, окно {period}. Бутстреп по МО", loc="left")
    fig.text(0.01, 0.01, f"Медиана {values.median():.2f}, уверенность 0.9 и выше у {(values >= 0.9).mean():.0%} МО. Серым цветом показаны МО без полных данных.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def robust_transitions(wide, moved, territories, palette, threshold=0.5):
    first, last = wide.columns[0], wide.columns[-1]
    changed = wide[wide[first] != wide[last]]
    table = pd.DataFrame({
        "МО": territories.loc[changed.index, "name"],
        "регион": territories.loc[changed.index, "region"],
        "из типа": changed[first].map(palette["names"]),
        "в тип": changed[last].map(palette["names"]),
        "доля прогонов со сменой типа": moved.reindex(changed.index),
    })
    return table[table["доля прогонов со сменой типа"] >= threshold].sort_values("доля прогонов со сменой типа", ascending=False)


def rank_table(features, types, period):
    rows, assigned = window_rows(features, types, period)
    return (rows[list(RULE_NAMES)].rank(pct=True) * 100).round(0), assigned.to_numpy()


def round_cut(value):
    if value == 0:
        return 0.0
    digits = 2 - int(np.floor(np.log10(abs(value))))
    return round(float(value), max(digits, 0))


def rule_conditions(table, quantiles):
    result = []
    for column in table.columns:
        for cut in sorted({round_cut(v) for v in table[column].quantile(quantiles)}):
            result.append((column, "<=", cut))
            result.append((column, ">", cut))
    return result


def rule_holds(table, condition):
    column, sign, cut = condition
    return (table[column] <= cut).to_numpy() if sign == "<=" else (table[column] > cut).to_numpy()


def rule_mask(table, rule):
    return np.logical_and.reduce([rule_holds(table, c) for c in rule])


def rule_score(mask, target):
    hit = (mask & target).sum()
    if hit == 0:
        return 0.0, 0.0, 0.0
    precision, recall = hit / mask.sum(), hit / target.sum()
    return 2 * precision * recall / (precision + recall), precision, recall


def rule_search(candidates, masks, target, beam, depth):
    front = sorted(((rule_score(masks[c], target)[0], (c,)) for c in candidates), reverse=True)[:beam]
    best = front[0]
    for _ in range(depth - 1):
        grown = {}
        for _, rule in front:
            current = np.logical_and.reduce([masks[c] for c in rule])
            used = {c[0] for c in rule}
            for c in candidates:
                if c[0] in used:
                    continue
                key = tuple(sorted(rule + (c,)))
                if key not in grown:
                    grown[key] = rule_score(current & masks[c], target)[0]
        front = sorted(((value, rule) for rule, value in grown.items()), reverse=True)[:beam]
        if front[0][0] > best[0] + 0.01:
            best = front[0]
    return best[1]


def rule_text(rule):
    return " и ".join(f"{RULE_NAMES[c]} в {'нижних' if s == '<=' else 'верхних'} {(v if s == '<=' else 100 - v):g}% МО" for c, s, v in rule)


def type_rules(features, types, palette, first, last, settings):
    table, labels = rank_table(features, types, last)
    check, check_labels = rank_table(features, types, first)
    candidates = rule_conditions(table, settings["quantiles"])
    masks = {c: rule_holds(table, c) for c in candidates}
    rows = []
    for t in palette["order"]:
        target = labels == t
        rule = rule_search(candidates, masks, target, settings["beam"], settings["depth"])
        f1, precision, recall = rule_score(rule_mask(table, rule), target)
        _, check_precision, check_recall = rule_score(rule_mask(check, rule), check_labels == t)
        rows.append({
            "номер типа": t, "тип": palette["names"][t], "правило": rule_text(rule), "МО": int(target.sum()),
            "precision": precision, "recall": recall, "F1": f1,
            f"precision в {first}": check_precision, f"recall в {first}": check_recall,
        })
    return pd.DataFrame(rows)


def nearest_capital(territories, ids):
    places = territories.loc[ids].dropna(subset=["lat", "lon"])
    capitals = territories[territories["capital"].astype(bool)].dropna(subset=["lat", "lon"])
    lat, lon = np.radians(places["lat"].to_numpy(float)), np.radians(places["lon"].to_numpy(float))
    clat, clon = np.radians(capitals["lat"].to_numpy(float)), np.radians(capitals["lon"].to_numpy(float))
    a = np.sin((lat[:, None] - clat[None, :]) / 2) ** 2 + np.cos(lat)[:, None] * np.cos(clat)[None, :] * np.sin((lon[:, None] - clon[None, :]) / 2) ** 2
    distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return pd.DataFrame({
        "км до столицы региона": distance.min(axis=1), "ближайшая столица": capitals["name"].to_numpy()[distance.argmin(axis=1)],
    }, index=places.index).round({"км до столицы региона": 1})


def describe(config):
    out = config["paths"]["results"]
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()
    data = load_data(config)
    features, territories = data["features"], data["territories"]
    original = pd.read_csv(config["paths"]["processed"] / "territories.csv", index_col=0)
    types = pd.read_csv(out / "types.csv")
    settings = config["describe"]
    palette = {
        "order": settings["order"],
        "names": {int(k): v for k, v in settings["names"].items()},
        "short": {int(k): v for k, v in settings["short_names"].items()},
        "colors": dict(zip(settings["order"], settings["colors"])),
    }
    periods = sorted(types["period"].unique())
    first, last = periods[0], periods[-1]
    wide = types.pivot(index="territory_id", columns="period", values="type").dropna().astype(int)
    distance = nearest_capital(territories, types["territory_id"].unique())
    rules = type_rules(features, types, palette, first, last, settings["rules"])
    rules.round(3).to_csv(out / "type_rules.csv", index=False)

    summary = summarize(features, types, last)
    means = profiles(features, types, last)
    delta = changes(features, types, first, last)
    shown, statistic, p_value = mobility_check(load_mobility(config, territories), types, last)
    names = pd.Series(palette["names"], name="название")
    summary.join(names).to_csv(out / "summary.csv")
    means.join(names).to_csv(out / "profiles.csv")
    examples(features, territories, types, last).join(names).to_csv(out / "examples.csv")
    delta.join(names).to_csv(out / "changes.csv")
    shown.groupby("type")["km"].agg(["size", "median"]).join(names).assign(kruskal_h=statistic, p_value=p_value).to_csv(out / "mobility_check.csv")

    _, assigned = window_rows(features, types, last)
    shapes = municipality_shapes(config, original)
    plot_map(shapes, summary, assigned, last, palette, figures / "types_map.png")
    if (out / "confidence.csv").exists():
        plot_confidence(shapes, pd.read_csv(out / "confidence.csv"), last, figures / "types_confidence.png")
        moved = pd.read_csv(out / "transition_robustness.csv", index_col="territory_id")["moved_share"]
        robust = robust_transitions(wide, moved, territories, palette).join(distance)
        robust.to_csv(out / "robust_transitions.csv")
        print(f"robust transitions: {len(robust)} of {int((wide[first] != wide[last]).sum())} municipalities that changed type")
        print(f"median km to the regional capital for reliable changes by target type, all municipalities {distance['км до столицы региона'].median():.0f}:")
        print(robust.groupby("в тип")["км до столицы региона"].agg(["size", "median"]).round(0).to_string())
    plot_alluvial(wide, palette, figures / "transitions_alluvial.png")
    plot_transitions(wide, palette, figures / "transitions_matrix.png")
    plot_profiles(means, palette, last, figures / "types_profiles.png")
    plot_changes(delta, palette, first, last, figures / "types_changes.png")
    plot_mobility(shown, statistic, p_value, palette, figures / "mobility_check.png")
    approaches = out / "dynamics_approaches.csv"
    if approaches.exists():
        plot_dynamics(pd.read_csv(approaches), figures / "dynamics_approaches.png")

    colors = {"background": "#f1f0ec", "edge": "#dcdbd5", "surface": SURFACE, "muted": MUTED}
    gap, r2 = spending_gap(features, territories, types, last)
    gap.assign(type=gap["type"].map(palette["names"])).round(4).to_csv(out / "spending_gap.csv")
    early, _ = spending_gap(features, territories, types, first)
    stability = gap_stability(gap, r2, early)
    stability.round(4).to_csv(out / "spending_gap_summary.csv", index=False)
    plot_gap(shapes, gap, r2, last, colors, figures / "spending_gap_map.png")
    monthly = pd.read_parquet(config["paths"]["processed"] / "spending_monthly.parquet")
    k = config["model"]["k"]
    shares, summary, seasonality = monthly_transitions(monthly, original, k)
    shares.round(4).to_csv(out / "monthly_changes.csv")
    summary.assign(**seasonality).round(4).to_csv(out / "monthly_summary.csv", index=False)
    plot_monthly(shares, k, seasonality["municipalities"], colors, figures / "monthly_changes.png")
    market, rubles, count = marketplaces(monthly, types, last)
    catch_up(market, rubles, palette["order"][0]).rename(index=palette["short"]).round(3).to_csv(out / "marketplaces.csv")
    plot_marketplaces(market, palette, count, last, colors, figures / "marketplaces.png")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.join(names).round(3).to_string())
    print(f"mobility check: H = {statistic:.1f}, p = {p_value:.2g}, {len(shown)} municipalities")
    print(f"spending level from the local economy: out-of-fold R2 {r2:.3f}; spend 15% or more above the forecast: {(gap['gap'] >= 0.15).sum()}, below: {(gap['gap'] <= -0.15).sum()}")
    print(f"spending gap rank correlation {first} vs {last}: Spearman {stability.at[0, 'spearman_first_last']:.3f} on {stability.at[0, 'municipalities']} municipalities")
    print(rules[["тип", "правило", "МО", "precision", "recall", "F1"]].round(2).to_string(index=False))
    print(summary.round(3).to_string(index=False))
    print({name: round(float(value), 3) for name, value in seasonality.items()})
    print(f"tables in {out}, figures in {figures}")