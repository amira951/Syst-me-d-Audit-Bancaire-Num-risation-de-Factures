"""
Apprentissage SUPERVISÉ : à partir d'exemples (texte → catégorie),
le modèle apprend une frontière de décision.

On compare volontairement 5 algo du cours d'IA :
  - Naive Bayes     : probabilités, hypothèse d'indépendance
  - LogReg          : modèle linéaire, coefficients = explication
  - SVM linéaire    : marge maximale
  - MLP             : premier réseau de neurones (1 couche cachée)
  - k-NN            : "dis-moi qui sont tes voisins"
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DIR_MODELS = ROOT / "models"
DIR_DATA = ROOT / "data"
DIR_MODELS.mkdir(exist_ok=True)
DIR_DATA.mkdir(exist_ok=True)
CHEMIN_MODELE = DIR_MODELS / "clf_categories.joblib"
CHEMIN_FEEDBACK = DIR_DATA / "feedback_categories.jsonl"

CATEGORIES = [
    "informatique",
    "logiciel",
    "mobilier",
    "services",
    "fournitures",
    "telecom",
    "transport",
    "energie",
    "autre",
]

# Jeu d'amorçage = "weak supervision" : labels faits à la main.
# En vrai projet bancaire, ça vient du plan comptable / d'un expert.
SEED: list[tuple[str, str]] = [
    # informatique
    ("dell precision workstation intel xeon quad core", "informatique"),
    ("fast dell windows 10 desktop computer core ram dvd", "informatique"),
    ("sama rgb gaming pc desktop nvidia gtx 1060", "informatique"),
    ("dell optiplex desktop computer i7 ram ssd", "informatique"),
    ("laptop macbook pro 16 inch m3", "informatique"),
    ("écran dell 27 pouces 4k usb-c", "informatique"),
    ("clavier mécanique logitech souris sans fil", "informatique"),
    ("disque dur ssd samsung 1to nvme", "informatique"),
    ("imprimante hp laserjet toner", "informatique"),
    ("serveur rack hp proliant ram ecc", "informatique"),
    ("carte graphique rtx nvidia workstation", "informatique"),
    ("ordinateur portable lenovo thinkpad", "informatique"),
    # logiciel
    ("microsoft windows 10 pro license", "logiciel"),
    ("adobe creative cloud annual subscription", "logiciel"),
    ("licence office 365 business", "logiciel"),
    ("antivirus bitdefender entreprise", "logiciel"),
    ("abonnement slack slack pro", "logiciel"),
    ("licence autodesk autocad", "logiciel"),
    ("saas crm salesforce subscription", "logiciel"),
    ("jetbrains intellij idea license", "logiciel"),
    ("microsoft azure devops seats", "logiciel"),
    ("oracle database license", "logiciel"),
    # mobilier
    ("chaise de bureau ergonomique herman miller", "mobilier"),
    ("bureau assis debout électrique", "mobilier"),
    ("armoire métallique dossiers", "mobilier"),
    ("table de réunion 8 places", "mobilier"),
    ("lampe de bureau led", "mobilier"),
    ("caisson tiroirs bois", "mobilier"),
    ("canapé d'accueil open space", "mobilier"),
    ("étagère bibliothèque bureau", "mobilier"),
    # services
    ("prestation conseil informatique", "services"),
    ("maintenance annuelle photocopieur", "services"),
    ("formation excel avancé", "services"),
    ("audit comptable honoraires", "services"),
    ("développement site web freelance", "services"),
    ("nettoyage locaux mensuel", "services"),
    ("support technique infogérance", "services"),
    ("prestation juridique contrat", "services"),
    ("consulting stratégie digitale", "services"),
    # fournitures
    ("ramette papier a4 80g", "fournitures"),
    ("stylos bic classeur agrafes", "fournitures"),
    ("cartouche encre canon", "fournitures"),
    ("cahiers post-it marqueurs", "fournitures"),
    ("chemises cartonnées archives", "fournitures"),
    ("ruban adhésif scotch", "fournitures"),
    ("toner compatible brother", "fournitures"),
    # telecom
    ("forfait mobile entreprise orange", "telecom"),
    ("abonnement fibre pro sosh", "telecom"),
    ("ligne téléphonique sip trunk", "telecom"),
    ("box internet professionnelle", "telecom"),
    ("cartes sim flotte mobile", "telecom"),
    ("hébergement nom de domaine", "telecom"),
    ("cloud storage google workspace", "telecom"),
    # transport
    ("billet train tgv paris lyon", "transport"),
    ("course taxi uber business", "transport"),
    ("location voiture hertz 3 jours", "transport"),
    ("frais de port chronopost", "transport"),
    ("péage autoroute essence trajet", "transport"),
    ("billet avion air france", "transport"),
    ("livraison express dhl", "transport"),
    # energie
    ("facture électricité edf professionnels", "energie"),
    ("gaz naturel engie locaux", "energie"),
    ("recharge véhicules électriques", "energie"),
    ("contrat fourniture électricité", "energie"),
    ("chauffage urbain copropriété", "energie"),
    # autre
    ("assurance responsabilité civile", "autre"),
    ("cotisation syndicat professionnel", "autre"),
    ("frais bancaires virement sepa", "autre"),
    ("don association mécénat", "autre"),
    ("restaurant d'entreprise repas", "autre"),
]


def _charger_feedback() -> list[tuple[str, str]]:
    if not CHEMIN_FEEDBACK.exists():
        return []
    extra = []
    for ligne in CHEMIN_FEEDBACK.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        obj = json.loads(ligne)
        extra.append((obj["texte"], obj["categorie"]))
    return extra


def jeu_complet() -> tuple[list[str], list[str]]:
    paires = SEED + _charger_feedback()
    x, y = zip(*paires)
    return list(x), list(y)


def _pipeline(clf) -> Pipeline:
    # TF-IDF : chaque texte devient un vecteur
    # tf = fréquence du mot dans le doc
    # idf = rareté du mot dans le corpus
    # un mot rare et présent pèse plus qu'un "the" / "de"
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),  # "windows 10" compte comme un terme
                    min_df=1,
                    max_features=4000,
                ),
            ),
            ("clf", clf),
        ]
    )


def _classifieurs() -> dict:
    return {
        "Naive Bayes": MultinomialNB(),
        "Régression logistique": LogisticRegression(max_iter=400, class_weight="balanced"),
        "SVM linéaire": LinearSVC(class_weight="balanced"),
        "MLP (réseau de neurones)": MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=500,
            random_state=42,
        ),
        "k-NN (k=3)": KNeighborsClassifier(n_neighbors=3, metric="cosine"),
    }


def entrainer(force: bool = False) -> Pipeline:
    if CHEMIN_MODELE.exists() and not force:
        return joblib.load(CHEMIN_MODELE)

    x, y = jeu_complet()
    pipe = _pipeline(
        LogisticRegression(max_iter=400, class_weight="balanced")
    )
    pipe.fit(x, y)
    joblib.dump(pipe, CHEMIN_MODELE)
    return pipe


def comparer_modeles() -> list[dict]:
    """Le tableau que tu montres au jury : on ne 'croit' pas un algo, on le mesure."""
    x, y = jeu_complet()
    # 20 % de test — jamais vus à l'entraînement (règle d'or du ML)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    lignes = []
    for nom, clf in _classifieurs().items():
        pipe = _pipeline(clf)
        pipe.fit(x_tr, y_tr)
        pred = pipe.predict(x_te)
        lignes.append(
            {
                "modèle": nom,
                "famille": {
                    "Naive Bayes": "probabiliste",
                    "Régression logistique": "linéaire",
                    "SVM linéaire": "marge max",
                    "MLP (réseau de neurones)": "réseau",
                    "k-NN (k=3)": "instance",
                }[nom],
                "accuracy": round(accuracy_score(y_te, pred), 3),
                "f1_macro": round(f1_score(y_te, pred, average="macro"), 3),
            }
        )
    return sorted(lignes, key=lambda r: r["f1_macro"], reverse=True)


def expliquer(texte: str, top: int = 6) -> dict:
    """
    XAI (eXplainable AI) : obligatoire en banque.
    On utilise le fait que LogReg est linéaire :
        score(classe) = somme ( tfidf(mot) * coefficient(mot, classe) )
    Donc on peut dire QUELS mots ont poussé la décision.
    """
    pipe = entrainer()
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    X = vec.transform([texte or ""])
    proba = clf.predict_proba(X)[0]
    classes = list(clf.classes_)
    idx = int(np.argmax(proba))
    categorie = classes[idx]

    noms = np.array(vec.get_feature_names_out())
    poids = X.toarray()[0] * clf.coef_[idx]
    ordre = np.argsort(poids)
    mots_pour = [
        {"mot": noms[i], "poids": round(float(poids[i]), 3)}
        for i in ordre[::-1][:top]
        if poids[i] > 0
    ]
    mots_contre = [
        {"mot": noms[i], "poids": round(float(poids[i]), 3)}
        for i in ordre[:top]
        if poids[i] < 0
    ]
    return {
        "categorie": categorie,
        "confiance": round(float(proba[idx]), 3),
        "probas": {c: round(float(p), 3) for c, p in zip(classes, proba)},
        "mots_pour": mots_pour,
        "mots_contre": mots_contre,
    }


def _montant(item: dict) -> float:
    for k in ("gross_worth", "net_worth"):
        v = item.get(k)
        if v in (None, ""):
            continue
        try:
            return float(str(v).replace(" ", "").replace(",", ".").replace("€", ""))
        except ValueError:
            pass
    return 0.0


def classifier_facture(items: list[dict], texte_global: str = "") -> dict:
    if not items:
        exp = expliquer(texte_global)
        return {"globale": exp, "lignes": [], "repartition": {}}

    lignes = []
    poids_cat: dict[str, float] = {c: 0.0 for c in CATEGORIES}
    for it in items:
        desc = it.get("description") or ""
        exp = expliquer(desc)
        montant = _montant(it)
        lignes.append(
            {
                "numero": it.get("numero"),
                "description": desc,
                "categorie": exp["categorie"],
                "confiance": exp["confiance"],
                "montant": montant,
                "mots_cles": ", ".join(m["mot"] for m in exp["mots_pour"][:3]),
            }
        )
        poids_cat[exp["categorie"]] += max(montant, 1.0)

    totale = sum(poids_cat.values()) or 1.0
    repartition = {
        c: round(100 * v / totale, 1) for c, v in poids_cat.items() if v > 0
    }
    dominante = max(repartition, key=repartition.get)
    globale = expliquer(
        " ".join(it.get("description") or "" for it in items) + " " + (texte_global or "")
    )
    globale["categorie"] = dominante
    globale["part_dominante"] = repartition[dominante]
    return {"globale": globale, "lignes": lignes, "repartition": repartition}


def ajouter_feedback(texte: str, categorie: str) -> None:
    if categorie not in CATEGORIES or not (texte or "").strip():
        raise ValueError("Feedback invalide")
    with CHEMIN_FEEDBACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"texte": texte.strip(), "categorie": categorie}, ensure_ascii=False) + "\n")


def reentrainer() -> dict:
    pipe = entrainer(force=True)
    x, y = jeu_complet()
    return {"nb_exemples": len(x), "nb_classes": len(set(y)), "modele": str(CHEMIN_MODELE)}