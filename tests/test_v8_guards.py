"""Les gardes V8 doivent LEVER, jamais produire un resultat plausible et faux.

Chaque cas ci-dessous a une version silencieuse qui tourne sans erreur et donne
un modele entraine sur la mauvaise chose. C'est le mode de defaillance le plus
couteux en science : on ne le decouvre qu'apres avoir interprete les resultats.
"""
import pytest
import torch

from st_cdgm.models.causal_rcn import RCNCell
from st_cdgm.models.regression_head import GraphToGridDecoder
from st_cdgm.models.skip_direct import ConditionalSkipBlock
from st_cdgm.models.bernoulli_gamma import BernoulliGammaHead
from st_cdgm.priors import load_edge_prior


def test_prior_reindexing_rejects_unknown_node_order():
    """Un ordre de noeuds qui ne correspond pas au prior donnerait une matrice
    permutee : le modele tirerait A_dag vers des aretes arbitraires."""
    prior = load_edge_prior()
    with pytest.raises(ValueError, match="ordre de n"):
        prior.matrix(node_order=list(prior.nodes)[:-1] + ["inconnu"])


def test_prior_lag0_requires_instantaneous_structure():
    """23 des 32 aretes du prior sont a lag 0. Les imposer a A(1) reviendrait a
    decreter decale d'un jour ce qu'on sait simultane."""
    prior = load_edge_prior()
    lag0 = prior.edge_list(lag=0)
    assert lag0, "le prior doit contenir des aretes contemporaines"
    lag1 = prior.edge_list(lag=1)
    assert lag1, "le prior doit contenir des aretes decalees"
    assert len(lag0) + len(lag1) == len(prior)


def test_bg_head_and_skip_block_are_mutually_exclusive():
    """Le melange convexe redistribuerait mu : on ne saurait plus si
    E[residu|y] = 0 tient, donc la decomposition a deux etages non plus."""
    from st_cdgm.training.training_loop import train_epoch_stage1

    cell = RCNCell(num_vars=3, hidden_dim=8, driver_dim=3)

    class _Runner:
        def __init__(self, c):
            self.cell = c

    dec = GraphToGridDecoder(d_model=8, hr_h=16, hr_w=16, intermediate_h=4,
                             intermediate_w=4, n_heads=2, refine_channels=8)
    with pytest.raises(ValueError, match="exclusifs"):
        train_epoch_stage1(
            encoder=torch.nn.Linear(1, 1), rcn_runner=_Runner(cell),
            regression_head=dec, optimizer=None, data_loader=[],
            device=torch.device("cpu"), epoch_idx=0,
            bg_head=BernoulliGammaHead(dec.feature_channels),
            skip_block=ConditionalSkipBlock(3, (16, 16)))


def test_decoder_rejects_token_count_inconsistent_with_lr_grid():
    """En mode spatial, un N != lr_h*lr_w produirait un reshape silencieusement
    faux au lieu d'une erreur."""
    dec = GraphToGridDecoder(d_model=8, hr_h=16, hr_w=16, intermediate_h=4,
                             intermediate_w=4, n_heads=2, refine_channels=8,
                             query_mode="spatial", lr_h=5, lr_w=6)
    dec(torch.randn(1, 2, 30, 8))                      # 5*6 = 30 : OK
    with pytest.raises(ValueError, match="grille"):
        dec(torch.randn(1, 2, 31, 8))


def test_free_nodes_builder_refuses_mixed_node_conventions():
    """Melanger les jeux de noeuds donnerait un q incoherent avec le prior C7
    et avec la carte du routage diagonal."""
    import numpy as np
    import xarray as xr

    from st_cdgm.models.graph_builder import HeteroGraphBuilder

    st = xr.Dataset(
        {"orog": (("lat", "lon"), np.zeros((8, 9), "float32"))},
        coords={"lat": np.arange(8.0), "lon": np.arange(9.0)})
    with pytest.raises(ValueError, match="free_nodes_v8"):
        HeteroGraphBuilder(lr_shape=(4, 5), hr_shape=(8, 9), static_dataset=st,
                           include_mid_layer=True, free_nodes_v8=True)
