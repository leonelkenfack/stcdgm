# >>> Cell 2b : approvisionnement des donnees - cherche puis TELECHARGE
# Les .nc d'entrainement (3 Go) sont gitignores : ils ne viennent pas du clone.
# Cette cellule rend le notebook autonome, comme la Cell 2B des runs precedents.
import glob
import urllib.request as _ureq
import urllib.error as _uerr

DATA_ROOT = Path(f"{DRIVE_ROOT}/data") if IN_COLAB else Path("data/raw")
for _sub in ("train", "static_predictors"):
    (DATA_ROOT / _sub).mkdir(parents=True, exist_ok=True)

# Depot Zenodo du jeu NIWA/CCAM utilise depuis le debut. ACCESS-CM2 seulement :
# les autres GCM (OOD) ne sont pas sur Zenodo et doivent etre deposes a la main.
ZENODO = {
    "predictor_ACCESS-CM2_hist.nc":
        "https://zenodo.org/records/10889046/files/predictor_ACCESS-CM2_hist.nc?download=1",
    "pr_ACCESS-CM2_hist.nc":
        "https://zenodo.org/records/10889046/files/pr_ACCESS-CM2_hist.nc?download=1",
}


def stream_dl(url, dest, retries=5, chunk=1024 * 1024):
    """Telechargement avec REPRISE (header Range) et backoff exponentiel.

    3 Go sur une session Colab : une coupure reseau est probable, pas
    exceptionnelle. Sans reprise il faudrait tout recommencer. Le fichier est
    ecrit en .part puis renomme : un fichier tronque ne peut pas etre pris pour
    un telechargement reussi au run suivant.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        already = part.stat().st_size if part.exists() else 0
        req = _ureq.Request(url)
        if already > 0:
            req.add_header("Range", f"bytes={already}-")
            print(f"    reprise a {already / 1e6:.1f} MB")
        try:
            with _ureq.urlopen(req, timeout=30) as resp:
                # Longueur ANNONCEE pour CETTE requete (206 -> ce qui reste).
                _cl = resp.headers.get("Content-Length")
                attendu = int(_cl) if _cl is not None else None
                with open(part, "ab" if already > 0 else "wb") as f:
                    got, last, last_b = already, time.time(), already
                    while True:
                        c = resp.read(chunk)
                        if not c:
                            break
                        f.write(c)
                        got += len(c)
                        if time.time() - last >= 5:
                            sp = (got - last_b) / (time.time() - last) / 1e6
                            print(f"    {got / 1e6:8.1f} MB  --  {sp:5.1f} MB/s")
                            last, last_b = time.time(), got
            # Un serveur qui ferme la connexion en cours de route ne leve PAS :
            # read() rend vide, la boucle sort normalement, et sans ce controle
            # le .part tronque serait promu comme un fichier complet. C'est le
            # mode de defaillance le plus couteux ici : open_dataset accepterait
            # un .nc mutile et rendrait des NaN plusieurs cellules plus loin.
            if attendu is not None and (got - already) != attendu:
                raise ConnectionError(
                    f"flux tronque : {got - already} octets recus sur "
                    f"{attendu} annonces (le .part est conserve pour la reprise)")
            os.replace(part, dest)
            print(f"  OK {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except (_uerr.HTTPError, _uerr.URLError, TimeoutError, ConnectionError) as e:
            wait = min(60, 2 ** attempt)
            print(f"  WARN {type(e).__name__}: {e} -- nouvel essai dans {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"echec du telechargement apres {retries} essais : {url}")


def find(patterns, roots):
    """Recherche recursive : le fichier peut etre range differemment sur Drive."""
    for r in roots:
        for pat in patterns:
            hits = sorted(glob.glob(str(Path(r) / "**" / pat), recursive=True))
            if hits:
                return Path(hits[0])
    return None


ROOTS = [Path("data/raw"), DATA_ROOT] + ([Path(DRIVE_ROOT)] if IN_COLAB else [])

LR_PATH = find(["predictor_ACCESS-CM2_hist.nc"], ROOTS)
HR_PATH = find(["pr_ACCESS-CM2_hist.nc"], ROOTS)
# Le statique EST suivi par git : il arrive avec le clone.
STATIC_PATH = find(["*NZ_Invariant.nc", "ERA5_eval_ccam_12km*.nc"], ROOTS)

if LR_PATH is None:
    print("[2b] predicteurs absents -> Zenodo")
    LR_PATH = stream_dl(ZENODO["predictor_ACCESS-CM2_hist.nc"],
                        DATA_ROOT / "train" / "predictor_ACCESS-CM2_hist.nc")
if HR_PATH is None:
    print("[2b] precipitation HR absente -> Zenodo (2,5 Go, comptez ~10 min)")
    HR_PATH = stream_dl(ZENODO["pr_ACCESS-CM2_hist.nc"],
                        DATA_ROOT / "train" / "pr_ACCESS-CM2_hist.nc")
if STATIC_PATH is None:
    raise FileNotFoundError(
        "Statique HR (orographie, masque terre/mer) introuvable. Il est suivi "
        "par git et devrait arriver avec le clone : verifier "
        "data/raw/static_predictors/ dans le depot.")

for _p in (LR_PATH, HR_PATH, STATIC_PATH):
    guard(_p)

# Verification de taille : un .nc tronque passerait open_dataset et donnerait
# des NaN silencieux plusieurs cellules plus loin.
for _n, _p, _min_mb in (("LR", LR_PATH, 500), ("HR", HR_PATH, 2000)):
    _mb = _p.stat().st_size / 1e6
    if _mb < _min_mb:
        raise RuntimeError(
            f"{_n} = {_p} ne fait que {_mb:.0f} MB (attendu > {_min_mb} MB) : "
            f"telechargement incomplet, supprimer le fichier et relancer.")

print(f"LR      : {LR_PATH}  ({LR_PATH.stat().st_size / 1e6:.0f} MB)")
print(f"HR      : {HR_PATH}  ({HR_PATH.stat().st_size / 1e6:.0f} MB)")
print(f"statique: {STATIC_PATH}")
