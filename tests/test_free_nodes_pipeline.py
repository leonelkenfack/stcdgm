"""Couture V8 : les 13 nœuds libres doivent traverser tout le chemin de données.

Le risque n'est pas qu'une formule soit fausse — ``derived.py`` a son propre
test — mais que les canaux arrivent dans le mauvais ORDRE au ``driver`` du RCN.
Le routage diagonal utilise par défaut la carte identité : le canal v va à la
variable v. Une permutation silencieuse donnerait un modèle qui tourne, dont
le DAG porterait des noms faux, et rien ne le signalerait.
"""
import numpy as np
import pytest
import xarray as xr

from st_cdgm.data.derived import FREE_NODES, RAW_VARS, free_nodes_dataset


@pytest.fixture
def raw_ds():
    rng = np.random.default_rng(0)
    T, H, W = 6, 23, 26
    lat = np.linspace(-59.4, -26.4, H)
    lon = np.linspace(150.6, 188.1, W)
    base = {850: (8.0, 285.0, 6e-3), 500: (18.0, 258.0, 1.2e-3),
            250: (35.0, 225.0, 6e-5)}
    data = {}
    for lev, (uu, tt, qq) in base.items():
        for pre, val, sd in (("u", uu, 2.0), ("v", 1.0, 2.0), ("w", 0.0, 0.05),
                             ("t", tt, 1.0), ("q", qq, qq * 0.2)):
            data[f"{pre}_{lev}"] = (("time", "lat", "lon"),
                                    np.abs(val + rng.normal(0, sd, (T, H, W))))
    return xr.Dataset(data, coords={"time": np.arange(T), "lat": lat, "lon": lon})


def test_dataset_carries_exactly_the_free_nodes(raw_ds):
    out = free_nodes_dataset(raw_ds)
    assert list(out.data_vars) == list(FREE_NODES)
    assert set(out.data_vars).isdisjoint(RAW_VARS)
    assert out[FREE_NODES[0]].dims == raw_ds[RAW_VARS[0]].dims


def test_channel_order_survives_to_array(raw_ds):
    """``_dataset_to_numpy`` empile via ``to_array``. L'ordre des canaux doit
    rester celui de FREE_NODES, sinon la carte identité du routage diagonal
    associe chaque variable au mauvais champ."""
    from st_cdgm.data.pipeline import _dataset_to_numpy

    out = free_nodes_dataset(raw_ds)
    arr = _dataset_to_numpy(out, "time", "lat", "lon")     # [T, C, H, W]
    assert arr.shape[1] == len(FREE_NODES)
    for c, name in enumerate(FREE_NODES):
        np.testing.assert_allclose(arr[:, c], out[name].values, rtol=1e-6)


def test_driver_reaches_the_matching_rcn_variable(raw_ds):
    """Bout en bout : perturber le champ du nœud i ne doit déplacer que la
    variable i de l'état du RCN."""
    import torch

    from st_cdgm.data.pipeline import _dataset_to_numpy
    from st_cdgm.models.causal_rcn import RCNCell

    out = free_nodes_dataset(raw_ds)
    arr = _dataset_to_numpy(out, "time", "lat", "lon")
    q = len(FREE_NODES)
    # [C, H, W] -> nodes [N, C], comme lr_grid_to_nodes
    grid = torch.from_numpy(arr[0]).float()
    driver = grid.reshape(q, -1).transpose(0, 1).contiguous()
    driver = (driver - driver.mean(0)) / (driver.std(0) + 1e-6)

    torch.manual_seed(0)
    cell = RCNCell(num_vars=q, hidden_dim=8, driver_dim=q,
                   driver_routing="diagonal").eval()
    H0 = torch.zeros(q, driver.shape[0], 8)
    with torch.no_grad():
        base, _, _ = cell(H0, driver)
        for i in (0, 6, q - 1):
            d = driver.clone()
            d[:, i] += 3.0
            moved = ((cell(H0, d)[0] - base).abs().mean(dim=(1, 2)) > 1e-6)
            assert moved.sum() == 1 and bool(moved[i]), (
                f"le canal de {FREE_NODES[i]} déplace {int(moved.sum())} "
                f"variable(s), et pas la sienne")
