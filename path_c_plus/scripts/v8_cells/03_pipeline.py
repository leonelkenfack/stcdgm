# >>> Cell 3 : pipeline - les 13 noeuds derives a la volee
from st_cdgm.data.pipeline import NetCDFDataPipeline

# lr_free_nodes=True : la derivation est faite UNE fois sur le jeu complet, pas
# par fenetre. Tout l'aval (_dataset_to_numpy, lr_grid_to_nodes, le driver du
# RCN) voit alors 13 canaux dans l'ordre de FREE_NODES, ce qui est exactement
# la carte identite du routage diagonal.
t0 = time.time()
pipeline = NetCDFDataPipeline(
    lr_path=guard(LR_PATH), hr_path=guard(HR_PATH), static_path=guard(STATIC_PATH),
    seq_len=SEQ_LEN,
    baseline_strategy=str(CONFIG.data.baseline_strategy),
    baseline_factor=int(CONFIG.data.baseline_factor),
    normalize=bool(CONFIG.data.normalize),
    nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
    precipitation_delta=float(CONFIG.data.precipitation_delta),
    lr_free_nodes=bool(V8.free_nodes),
    lr_variables=None if V8.free_nodes else list(CONFIG.data.lr_variables),
    hr_variables=list(CONFIG.data.hr_variables),
    static_variables=(list(CONFIG.data.static_variables)
                      if CONFIG.data.get("static_variables") else []),
    train_start_date=CONFIG.data.get("train_start_date"),
    train_end_date=CONFIG.data.get("train_end_date"),
    val_start_date=CONFIG.data.get("val_start_date"),
    val_end_date=CONFIG.data.get("val_end_date"),
    test_start_date=CONFIG.data.get("test_start_date"),
    test_end_date=CONFIG.data.get("test_end_date"),
)
LR_VARS = list(pipeline.get_lr_dataset().data_vars)
print(f"[{time.time() - t0:.0f}s] canaux LR ({len(LR_VARS)}) : {LR_VARS}")

if V8.free_nodes:
    # L'ordre est load-bearing : une permutation silencieuse ferait tourner le
    # modele en associant chaque variable au mauvais champ, sans rien signaler.
    assert LR_VARS == list(FREE_NODES), (
        f"ordre des canaux != FREE_NODES\n  recu   : {LR_VARS}\n  attendu: {list(FREE_NODES)}")

# stride : sans lui build_sequence_dataset retombe sur 1, soit 10 935
# echantillons par epoque au lieu des ~2 734 configures. Quatre fois le temps
# de calcul ET quatre fois le cache de l'etage 2 - c'est ce qui fait passer
# l'empreinte memoire de 1,1 Go a 4,4 Go, donc d'un run qui tient sur T4 a un
# run qui sature.
# Le split de TEST prend le stride des references (Cell 2) : les metriques
# auxquelles on se compare ont ete calculees sur ces fenetres-la.
train_dataset = pipeline.build_sequence_dataset(split="train", stride=STRIDE_TRAIN,
                                                training=True)
val_dataset   = pipeline.build_sequence_dataset(split="val", stride=STRIDE_TRAIN)
test_dataset  = pipeline.build_sequence_dataset(split="test", stride=STRIDE_EVAL)
print(f"stride : train/val={STRIDE_TRAIN} | test={STRIDE_EVAL}")

_s = next(iter(train_dataset))
print("echantillon : lr", tuple(_s["lr"].shape),
      "| residual", tuple(_s["residual"].shape),
      "| baseline", tuple(_s["baseline"].shape))
assert _s["lr"].shape[1] == len(LR_VARS)
