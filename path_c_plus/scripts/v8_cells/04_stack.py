# >>> Cell 4 : la pile V8
from st_cdgm.models.graph_builder import HeteroGraphBuilder
from st_cdgm.models.intelligible_encoder import (
    IntelligibleVariableEncoder, IntelligibleVariableConfig)
from st_cdgm.models.causal_rcn import RCNCell, RCNSequenceRunner
from st_cdgm.models.regression_head import GraphToGridDecoder
from st_cdgm.models.bernoulli_gamma import BernoulliGammaHead
from st_cdgm.priors import load_edge_prior, DEFAULT_LEVEL_FLOORS

torch.manual_seed(SEED)

# --- graphe : 13 types de noeuds REELLEMENT distincts ----------------------
# free_nodes_v8 remplace entierement le jeu de noeuds (pas de GP850/500/250).
builder = HeteroGraphBuilder(
    lr_shape=LR_SHAPE, hr_shape=HR_SHAPE,
    static_dataset=pipeline.get_static_dataset(),
    include_mid_layer=not bool(V8.free_nodes),
    free_nodes_v8=bool(V8.free_nodes),
)
hetero_template, _report = builder.build()
NODE_TYPES = list(builder.dynamic_node_types)
Q = len(NODE_TYPES)
print(f"q = {Q} variables : {NODE_TYPES}")

# --- encodeur : un metachemin spatial par variable -------------------------
enc_cfgs = [IntelligibleVariableConfig(name=f"{n}_spat",
                                       meta_path=(n, "spat_adj", n), pool="mean")
            for n in NODE_TYPES]
encoder = IntelligibleVariableEncoder(
    configs=enc_cfgs, hidden_dim=HIDDEN,
    conditioning_dim=int(CONFIG.encoder.conditioning_dim)).to(DEVICE)

# MATERIALISATION OBLIGATOIRE. L'encodeur est bati sur des LazyModule : ses
# poids n'existent qu'apres un premier forward, et il faut un graphe PORTANT
# DES FEATURES (le template n'en a pas). Les notebooks precedents ne s'en
# apercevaient pas parce qu'ils chargeaient un checkpoint - load_state_dict
# materialise. V8 part de zero : sans ce forward a blanc, encoder.parameters()
# serait VIDE au moment de construire l'optimiseur, et l'encodeur resterait a
# son initialisation aleatoire pendant tout l'entrainement, EN SILENCE.
from st_cdgm.evaluation.evaluation_xai import convert_sample_to_batch

def convert_sample_v8(sample, bld=None, dev=None):
    """Comme convert_sample_to_batch, mais chaque noeud recoit SON canal.

    Le convertisseur partage fait :
        dynamic_features = {nt: lr_nodes[0] for nt in dynamic_node_types}
    c'est-a-dire le MEME tenseur 13 canaux pour les 13 types. L'etat initial
    H(0) ne distingue donc les variables que par les poids de convolution de
    leur metachemin - exactement le defaut que V8 corrige ailleurs. Le routage
    diagonal V7-M2 ne couvre que la RECURRENCE ; sans ce correctif, le point de
    depart de la recurrence reste indifferencie.

    Ici le noeud d'indice v recoit uniquement le canal v, dans l'ordre de
    FREE_NODES - la meme carte identite que le routage diagonal.
    """
    bld = bld if bld is not None else builder
    dev = dev if dev is not None else DEVICE
    lr_seq = sample["lr"]
    steps = [bld.lr_grid_to_nodes(lr_seq[t]) for t in range(lr_seq.shape[0])]
    if V8.free_nodes:
        first = steps[0]                                   # [N_lr, 13]
        feats = {nt: first[:, i:i + 1] for i, nt in enumerate(NODE_TYPES)}
    else:
        feats = {nt: steps[0] for nt in bld.dynamic_node_types}
    return {"lr": torch.stack(steps, dim=0),
            "residual": sample["residual"], "baseline": sample.get("baseline"),
            "hetero": bld.prepare_step_data(feats).to(dev)}

with torch.no_grad():
    _warm = convert_sample_v8(_s, builder, DEVICE)
    _H0 = encoder.init_state(_warm["hetero"])
_n_enc = sum(p.numel() for p in encoder.parameters())
assert _n_enc > 0, "encodeur non materialise : l'optimiseur serait vide"
assert _H0.shape[0] == Q, f"H_init a {_H0.shape[0]} variables, attendu {Q}"
print(f"encodeur  : {_n_enc:,} parametres materialises | H_init {tuple(_H0.shape)}")

# --- prior C7 --------------------------------------------------------------
edge_prior, A_prior, A_inst_prior = None, None, None
if V8.edge_prior and V8.free_nodes:
    edge_prior = load_edge_prior()
    assert len(edge_prior.nodes) == Q, (
        f"prior sur {len(edge_prior.nodes)} noeuds, graphe a {Q} variables")
    # node_order : l'ordre du RCN n'a aucune raison d'etre celui du YAML.
    # FILTRAGE PAR LAG obligatoire : A_dag est A(1), A_inst est A(0). Initialiser
    # A(1) avec TOUTES les aretes y injecterait les 24 aretes contemporaines que
    # la perte route pourtant vers A(0) — et A(0) demarrerait au bruit, sans
    # prior, alors que c'est la structure que C2/C5 existe pour exprimer.
    A_prior = torch.as_tensor(edge_prior.matrix(lag=1, node_order=NODE_TYPES))
    A_inst_prior = (torch.as_tensor(edge_prior.matrix(lag=0, node_order=NODE_TYPES))
                    if V8.instantaneous else None)
    print(f"prior C7  : {len(edge_prior)} aretes {edge_prior.by_level()}")
    print(f"            orientation {edge_prior.by_orient()}")
    print(f"            init : A(1) <- {len(edge_prior.edge_list(lag=1))} aretes | "
          f"A(0) <- {len(edge_prior.edge_list(lag=0))} aretes")
    if not V8.instantaneous and edge_prior.edge_list(lag=0):
        raise ValueError(
            f"{len(edge_prior.edge_list(lag=0))} aretes du prior sont a lag 0 "
            f"mais V8.instantaneous=False : elles seraient ecrasees sur A(1).")

# --- RCN : routage diagonal (V7-M2) + A(0) contemporaine -------------------
rcn_cell = RCNCell(
    num_vars=Q, hidden_dim=HIDDEN, driver_dim=len(LR_VARS),
    reconstruction_dim=len(LR_VARS),
    dropout=float(CONFIG.rcn.dropout),
    dag_prior=A_prior, inst_prior=A_inst_prior,
    instantaneous=bool(V8.instantaneous),
    driver_routing=("diagonal" if (V8.diagonal_driver and V8.free_nodes) else "shared"),
).to(DEVICE)
rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get("detach_interval"))
print(f"RCN       : routage={rcn_cell.driver_routing} | "
      f"A(0)={'oui' if rcn_cell.A_inst is not None else 'non'}")

# --- decodeur : requetes spatiales (A2a) -----------------------------------
_rh = CONFIG.two_stage.regression_head
regression_head = GraphToGridDecoder(
    d_model=HIDDEN, hr_h=HR_SHAPE[0], hr_w=HR_SHAPE[1],
    intermediate_h=int(_rh.intermediate_h), intermediate_w=int(_rh.intermediate_w),
    n_heads=int(_rh.n_heads), refine_channels=int(_rh.refine_channels),
    query_mode=("spatial" if V8.spatial_queries else "learned"),
    lr_h=LR_SHAPE[0], lr_w=LR_SHAPE[1],
).to(DEVICE)
print(f"decodeur  : query_mode={regression_head.query_mode} | "
      f"features={regression_head.feature_channels}")

# --- tete Bernoulli-Gamma (A3) --------------------------------------------
bg_head = (BernoulliGammaHead(regression_head.feature_channels).to(DEVICE)
           if V8.bernoulli_gamma else None)
# A3 exclut le melange convexe du skip-block : A4 propose de le retirer, et il
# casserait l'interpretation de mu = p*alpha*beta comme moyenne conditionnelle.
skip_block = None

STAGE1_MODULES = [encoder, rcn_cell, regression_head] + ([bg_head] if bg_head else [])
n_par = sum(p.numel() for m in STAGE1_MODULES for p in m.parameters())
print(f"parametres etage 1 : {n_par:,}")
