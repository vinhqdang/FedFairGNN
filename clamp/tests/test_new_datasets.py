"""Loader checks for NBA, Facebook 107, AMiner-S and pokec_n (skipped if the raw files are missing)."""
import os

import pandas as pd
import pytest
import torch

from src.data.datasets import load_dataset

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RAW = os.path.join(ROOT, "raw")

REQUIRED = {
    "nba": ["nba/nba.csv", "nba/nba_relationship.txt"],
    "facebook": ["facebook/107.edges", "facebook/107.feat", "facebook/107.featnames"],
    "aminer_s": ["aminer_s/edgelist_Small.txt", "aminer_s/labels_Small.txt",
                 "aminer_s/sens_Small.txt", "aminer_s/X_Small.npz"],
    "pokec_n": ["pokec/region_job_2.csv", "pokec/region_job_2_relationship.txt"],
}


def _present(name):
    return all(os.path.exists(os.path.join(RAW, f)) for f in REQUIRED[name])


def _skip(name):
    return pytest.mark.skipif(not _present(name), reason=f"raw files for {name} missing")


_cache = {}


def _load(name):
    if name not in _cache:
        _cache[name] = load_dataset(name, root=ROOT, seed=0)
    return _cache[name]


@pytest.mark.parametrize("name", ["nba", "facebook", "aminer_s", "pokec_n"])
def test_shapes_and_edges(name):
    if not _present(name):
        pytest.skip(f"raw files for {name} missing")
    d = _load(name)
    n = d.num_nodes
    assert d.x.shape[0] == n == d.y.shape[0] == d.sensitive_attr.shape[0]
    assert d.x.dtype == torch.float32
    ei = d.edge_index
    assert int(ei.max()) < n and int(ei.min()) >= 0
    assert not bool((ei[0] == ei[1]).any()), "self-loops must be removed"
    fwd = set(map(tuple, ei.t().tolist()))
    assert fwd == set((b, a) for a, b in fwd), "edge list must be symmetric"


@pytest.mark.parametrize("name", ["nba", "facebook", "aminer_s", "pokec_n"])
def test_sensitive_attribute_not_in_features(name):
    if not _present(name):
        pytest.skip(f"raw files for {name} missing")
    d = _load(name)
    s = d.sensitive_attr.float()
    for c in range(d.x.shape[1]):
        assert not torch.equal(d.x[:, c], s), f"feature column {c} equals s"


@pytest.mark.parametrize("name", ["nba", "facebook", "aminer_s", "pokec_n"])
def test_labels_and_attribute_are_binary(name):
    if not _present(name):
        pytest.skip(f"raw files for {name} missing")
    d = _load(name)
    lab = d.train_mask | d.val_mask | d.test_mask
    assert set(torch.unique(d.sensitive_attr).tolist()) <= {0, 1}
    assert set(torch.unique(d.y[lab]).tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("name", ["nba", "facebook", "aminer_s", "pokec_n"])
def test_masks_are_disjoint(name):
    if not _present(name):
        pytest.skip(f"raw files for {name} missing")
    d = _load(name)
    tr, va, te = d.train_mask, d.val_mask, d.test_mask
    assert not bool((tr & va).any()) and not bool((tr & te).any()) and not bool((va & te).any())


@_skip("nba")
def test_nba_unlabelled_nodes_are_in_no_split():
    d = _load("nba")
    df = pd.read_csv(os.path.join(RAW, "nba", "nba.csv"))
    labelled = int((df["SALARY"] >= 0).sum())
    assert int((d.train_mask | d.val_mask | d.test_mask).sum()) == labelled
    unlabelled = torch.tensor((df["SALARY"] < 0).values)
    assert not bool((d.train_mask | d.val_mask | d.test_mask)[unlabelled].any())
    assert d.num_nodes == len(df)


@_skip("nba")
def test_nba_feature_count_drops_id_label_and_attribute():
    d = _load("nba")
    df = pd.read_csv(os.path.join(RAW, "nba", "nba.csv"))
    assert d.x.shape[1] == df.shape[1] - 3


@_skip("facebook")
def test_facebook_feature_count_drops_id_label_and_both_gender_columns():
    d = _load("facebook")
    # 1045 rows; 577 raw columns = id + 576 features; drop 220, 264, 265
    assert d.x.shape == (1045, 573)
