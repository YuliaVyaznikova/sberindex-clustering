import copy
import itertools
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .dynamics import refine_over_time, supra
from .methods import fuse
from .metrics import graph_scores
from .network import load_data, snapshot


def windows(config):
    data = load_data(config)
    periods = sorted(data["features"]["period"].unique())
    return periods, {p: snapshot(config, data, p) for p in periods}


def model(shots, periods, config, alpha, k, coupling, keep=None):
    ids, xs, graphs, affinities = [], [], [], []
    for p in periods:
        shot = shots[p]
        mask = np.ones(len(shot["ids"]), dtype=bool) if keep is None else np.isin(shot["ids"], keep)
        ids.append(shot["ids"][mask])
        xs.append(shot["x"][mask])
        graphs.append(shot["graph"][mask][:, mask].tocsr())
        affinities.append(fuse(shot["graph"], shot["attribute_graph"], alpha)[mask][:, mask].tocsr())
    labels = supra(affinities, ids, k, coupling)
    labels = refine_over_time(xs, graphs, ids, labels, k, config["model"]["refine_weight"], config["dynamics"]["smoothing"])
    frame = pd.concat([pd.Series(l, index=pd.Index(i, name="territory_id"), name=p) for p, i, l in zip(periods, ids, labels)], axis=1)
    return frame.rename_axis(columns="period")


def summarize(frame, shots, periods, reference):
    first, last = periods[0], periods[-1]
    both = frame[[first, last]].dropna()
    row = {"changed_first_last": float((both[first] != both[last]).mean())}
    for period in [first, last]:
        common = frame[period].dropna().index.intersection(reference[period].dropna().index)
        row[f"ari_reference_{period}"] = adjusted_rand_score(reference.loc[common, period], frame.loc[common, period])
    labels = [frame.loc[shots[p]["ids"], p].to_numpy().astype(int) for p in periods]
    row["SW"] = float(np.mean([silhouette_score(shots[p]["x"], l) for p, l in zip(periods, labels)]))
    row["MQ"] = float(np.mean([graph_scores(shots[p]["graph"], l)["MQ"] for p, l in zip(periods, labels)]))
    return row


def sensitivity(config, reference, out):
    grid = config["robustness"]["grid"]
    rows = []
    for graph_k in grid["graph_k"]:
        local = copy.deepcopy(config)
        local["network"]["k"] = graph_k
        periods, shots = windows(local)
        for alpha, k, coupling in itertools.product(grid["alpha"], grid["k"], grid["coupling"]):
            frame = model(shots, periods, config, alpha, k, coupling)
            rows.append({"graph_k": graph_k, "alpha": alpha, "k": k, "coupling": coupling, **summarize(frame, shots, periods, reference)})
            pd.DataFrame(rows).to_csv(out / "sensitivity.csv", index=False)
        print(time.strftime("%H:%M:%S"), f"sensitivity: graph k {graph_k} done, {len(rows)} configurations", flush=True)


def align(reference, labels, k):
    overlap = np.array([[np.sum((reference == i) & (labels == j)) for j in range(k)] for i in range(k)])
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.empty(k, dtype=int)
    mapping[cols] = rows
    return mapping[labels]


def bootstrap(config, shots, periods, reference, out):
    settings, k = config["robustness"], config["model"]["k"]
    everyone = np.array(sorted(set().union(*[set(shots[p]["ids"]) for p in periods])))
    stacked = reference.stack().dropna().astype(int)
    rng = np.random.default_rng(settings["seed"])
    frames = []
    for run in range(settings["runs"]):
        keep = rng.choice(everyone, int(settings["share"] * len(everyone)), replace=False)
        labels = model(shots, periods, config, config["model"]["alpha"], k, config["dynamics"]["coupling"], keep).stack().dropna().astype(int)
        common = labels.index.intersection(stacked.index)
        frames.append(pd.DataFrame({"run": run, "type": align(stacked.loc[common].to_numpy(), labels.loc[common].to_numpy(), k)}, index=common))
        if (run + 1) % 25 == 0:
            print(time.strftime("%H:%M:%S"), f"bootstrap: {run + 1} of {settings['runs']}", flush=True)
    runs = pd.concat(frames).reset_index()
    merged = runs.merge(stacked.rename("reference").reset_index(), on=["territory_id", "period"])
    confidence = merged.assign(same=merged["type"] == merged["reference"]).groupby(["territory_id", "period"])["same"].agg(["mean", "size"])
    confidence.columns = ["confidence", "runs"]
    confidence.to_csv(out / "confidence.csv")
    first, last = periods[0], periods[-1]
    wide = runs.pivot_table(index=["territory_id", "run"], columns="period", values="type").dropna(subset=[first, last])
    moved = (wide[first] != wide[last]).groupby(level="territory_id").agg(["mean", "size"])
    moved.columns = ["moved_share", "runs"]
    moved.to_csv(out / "transition_robustness.csv")


def robustness(config):
    out = config["paths"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    periods, shots = windows(config)
    reference = model(shots, periods, config, config["model"]["alpha"], config["model"]["k"], config["dynamics"]["coupling"])
    sensitivity(config, reference, out)
    bootstrap(config, shots, periods, reference, out)
    confidence = pd.read_csv(out / "confidence.csv")
    moved = pd.read_csv(out / "transition_robustness.csv")
    print(f"median confidence {confidence['confidence'].median():.2f}, share of municipality windows with confidence >= 0.9: {(confidence['confidence'] >= 0.9).mean():.1%}")
    print(f"municipalities that change type in at least half of the runs: {(moved['moved_share'] >= 0.5).sum()} of {len(moved)}")