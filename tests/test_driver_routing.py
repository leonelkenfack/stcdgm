"""V7-M2 — routage diagonal des drivers.

La propriété testée n'est pas numérique mais STRUCTURELLE : en mode partagé,
les q variables d'état reçoivent le même embedding de driver et ne diffèrent
que par un biais additif appris (``var_embed``). Elles sont donc
interchangeables à une permutation de paramètres près, et l'information
inter-variable circule hors du DAG. En mode diagonal, chaque variable ne voit
que ses propres canaux : le seul chemin restant est ``A[u, v]``.
"""
import torch

from st_cdgm.models.causal_rcn import RCNCell

Q, N, HID = 13, 32, 16


def _sensitivity(routing):
    """Nombre de variables affectées quand on perturbe UN canal de driver."""
    torch.manual_seed(0)
    cell = RCNCell(num_vars=Q, hidden_dim=HID, driver_dim=Q,
                   driver_routing=routing).eval()
    H = torch.zeros(Q, N, HID)
    d0 = torch.randn(N, Q)
    with torch.no_grad():
        base, _, _ = cell(H, d0)
        hits = []
        for v in range(Q):
            d1 = d0.clone()
            d1[:, v] += 5.0
            out, _, _ = cell(H, d1)
            delta = (out - base).abs().mean(dim=(1, 2))
            hits.append(int((delta > 1e-6).sum()))
    return cell, hits


def test_shared_routing_leaks_to_every_variable():
    """Témoin : sans routage diagonal, un canal touche TOUTES les variables."""
    _, hits = _sensitivity("shared")
    assert set(hits) == {Q}


def test_diagonal_routing_confines_each_channel():
    """Le cœur de V7-M2 : un canal ne touche que sa variable au premier pas."""
    _, hits = _sensitivity("diagonal")
    assert set(hits) == {1}


def test_per_variable_encoders_are_distinct():
    """Des poids partagés rendraient les variables à nouveau interchangeables."""
    cell, _ = _sensitivity("diagonal")
    assert cell.driver_W.shape == (Q, 1, HID)
    assert not torch.allclose(cell.driver_W[0], cell.driver_W[1])


def test_explicit_channel_map_supports_several_channels():
    cmap = [[i, (i + 1) % 15] for i in range(Q)]
    cell = RCNCell(num_vars=Q, hidden_dim=HID, driver_dim=15,
                   driver_routing="diagonal", driver_channel_map=cmap)
    assert cell.driver_W.shape == (Q, 2, HID)


def test_invalid_channel_maps_are_rejected():
    """Taille, largeur inégale et canaux hors bornes doivent lever, pas passer :
    une carte muette produirait un routage faux et plausible."""
    for bad in ([[0]] * (Q - 1),
                [[0, 1]] * (Q - 1) + [[0]],
                [[99]] * Q):
        try:
            RCNCell(num_vars=Q, hidden_dim=HID, driver_dim=15,
                    driver_routing="diagonal", driver_channel_map=bad)
        except ValueError:
            continue
        raise AssertionError(f"carte invalide acceptée : {bad[:2]}…")


def test_routing_map_survives_state_dict_roundtrip():
    """Sans persistance, un checkpoint rechargerait un routage différent
    sans rien signaler."""
    cell, _ = _sensitivity("diagonal")
    sd = cell.state_dict()
    assert "driver_channel_idx" in sd
    other = RCNCell(num_vars=Q, hidden_dim=HID, driver_dim=Q,
                    driver_routing="diagonal")
    other.load_state_dict(sd)
    assert torch.equal(other.driver_channel_idx, cell.driver_channel_idx)
