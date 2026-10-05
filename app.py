from datetime import datetime
import io
import json
import os
import re
import sqlite3
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st

from src.preprocessing import nettoyer_image_depuis_chemin
from src.ocr import extraire_texte
from src.extraction import extraire_infos_facture, extraire_items
from src.table_extraction import (
    extraire_seller_client_par_position,
    parser_bloc_entite,
)
from src.database import (
    initialiser_base,
    enregistrer_facture,
    recuperer_toutes_factures,
)
from src.llm_extraction import (
    extraire_lignes_facture,
    valider_lignes,
    statut_ollama,
    appeler_llm_texte,
    _to_float,
)
from src.ia_confiance import valider_iban


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FinTech Banking Audit & Analytics — PFE",
    layout="wide",
    page_icon="🏦",
)

initialiser_base()
llm_status = statut_ollama()


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
    .kpi-card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        border-left: 7px solid #2563eb;
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 13px;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .kpi-value {
        font-size: 28px;
        color: #0f172a;
        font-weight: 800;
        margin-top: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# OUTILS
# ============================================================

def parse_date_soigneux(date_str):
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()

    formats = [
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d.%m.%Y",
        "%Y.%m.%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass

    return None


def valeur_facture(f, *cles):
    for cle in cles:
        v = f.get(cle)
        if v not in (None, "", "None", "—"):
            return v
    return None


def kpi_card(titre, valeur):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{titre}</div>
            <div class="kpi-value">{valeur}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BASE DE DONNÉES
# ============================================================

def liste_factures():
    brut = recuperer_toutes_factures()

    if brut is None:
        return []

    if hasattr(brut, "to_dict"):
        return brut.to_dict("records")

    return list(brut)


def liste_factures_completes():
    try:
        conn = sqlite3.connect("factures.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT rowid AS _rowid, * FROM factures ORDER BY id DESC")
        factures_rows = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT * FROM items")
        items_rows = [dict(r) for r in cursor.fetchall()]

        conn.close()

        for f in factures_rows:
            f_id = f.get("id") or f.get("_rowid")
            f_num = f.get("invoice_no") or f.get("numero")

            f_items = []

            for it in items_rows:
                it_fid = it.get("facture_id") or it.get("facture_no")

                if str(it_fid) in [str(f_id), str(f_num)]:
                    f_items.append(
                        {
                            "numero": it.get("numero"),
                            "description": it.get("description"),
                            "qty": it.get("qty") or it.get("quantite"),
                            "net_price": it.get("net_price")
                            or it.get("prix_unitaire"),
                            "net_worth": it.get("net_worth")
                            or it.get("montant_ht"),
                            "gross_worth": it.get("gross_worth")
                            or it.get("montant_ttc"),
                        }
                    )

            f["items"] = f_items

        return factures_rows

    except Exception:
        return liste_factures()


def vider_historique():
    try:
        if os.path.exists("factures.db"):
            os.remove("factures.db")

        initialiser_base()
        return True

    except Exception:
        try:
            conn = sqlite3.connect("factures.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM items")
            cursor.execute("DELETE FROM factures")
            conn.commit()
            conn.close()
            return True

        except Exception:
            return False


# ============================================================
# EXTRACTION TABLEAU
# ============================================================

def extraire_tableau_rapide(texte_corrige):
    erreur = None

    if llm_status["disponible"]:
        try:
            items = extraire_lignes_facture(texte_corrige)

            if items:
                return valider_lignes(items), "IA (Qwen 2.5)", None

        except Exception as e:
            erreur = str(e)

    else:
        erreur = "Ollama indisponible"

    try:
        items = valider_lignes(extraire_items(texte_corrige) or [])
    except Exception:
        items = []

    return items, "Méthode classique", erreur


# ============================================================
# AUDIT BANCAIRE
# ============================================================

def auditer_facture_bancaire(infos, items, factures_existantes):
    alertes = []
    score = 100

    # 1. IBAN
    iban = infos.get("iban")
    iban_check = valider_iban(iban)

    if not iban:
        alertes.append("⚠️ Code IBAN absent sur la facture")
        score -= 20

    elif not iban_check.get("valide"):
        alertes.append(
            "🚨 IBAN invalide ou fictif : échec du contrôle mathématique Modulo 97"
        )
        score -= 40

    # 2. Doublons
    inv_no = str(infos.get("invoice_no") or "").strip()
    vendeur = str(infos.get("seller_name") or "").strip().lower()
    montant = _to_float(infos.get("gross_worth_total"))

    for f in factures_existantes:
        f_num = str(f.get("invoice_no") or "").strip()
        f_vendeur = str(f.get("seller_name") or "").strip().lower()
        f_montant = _to_float(f.get("gross_worth_total"))

        if inv_no and inv_no == f_num and inv_no not in ("—", "", "None"):
            alertes.append(
                f"🚨 Doublon détecté : la facture n°{inv_no} existe déjà en base"
            )
            score -= 50
            break

        if vendeur and vendeur == f_vendeur and montant and f_montant == montant:
            alertes.append(
                "⚠️ Suspicion de doublon : même vendeur et même montant déjà enregistrés"
            )
            score -= 30
            break

    # 3. Cohérence articles
    calculs_ko = sum(1 for i in items if not i.get("valide", True))

    if calculs_ko > 0:
        alertes.append(
            f"⚠️ {calculs_ko} article(s) présentent un écart de calcul"
        )
        score -= 15 * calculs_ko

    score = max(0, score)

    if score >= 80:
        statut = "CONFORME — Paiement autorisé"
        decision = "🟢 Auto-approuvé"

    elif score >= 50:
        statut = "RISQUE MOYEN — Vérification manuelle recommandée"
        decision = "🟠 Revue manuelle"

    else:
        statut = "RISQUE ÉLEVÉ — Paiement bloqué"
        decision = "🔴 Alerte / Rejet"

    return {
        "score": score,
        "statut_long": statut,
        "decision": decision,
        "alertes": alertes,
    }


# ============================================================
# EXPORT EXCEL
# ============================================================

def generer_excel_comptable(factures):
    output = io.BytesIO()
    df = pd.DataFrame(factures)

    colonnes = {
        "invoice_no": "N° Facture",
        "date": "Date Émission",
        "seller_name": "Nom Vendeur",
        "seller_address": "Adresse Vendeur",
        "client_name": "Nom Client",
        "client_address": "Adresse Client",
        "iban": "IBAN",
        "net_worth_total": "Total HT (€)",
        "vat_total": "Total TVA (€)",
        "gross_worth_total": "Total TTC (€)",
    }

    cols = [c for c in colonnes if c in df.columns]
    df_export = df[cols].rename(columns=colonnes)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Journal_Achats")

    return output.getvalue()


# ============================================================
# ASSISTANT
# ============================================================

def repondre_assistant(question, factures):
    q = question.lower().strip()

    if not factures:
        return "Aucune facture enregistrée dans la base."

    # Recherche produits / articles
    mots_produits = [
        "dell",
        "hp",
        "lenovo",
        "asus",
        "pc",
        "ordinateur",
        "gaming",
        "screen",
        "ecran",
        "écran",
        "desktop",
        "laptop",
    ]

    mot_trouve = next((m for m in mots_produits if m in q), None)

    if mot_trouve or any(k in q for k in ["contenant", "contient", "article", "produit"]):
        terme = mot_trouve if mot_trouve else q
        resultats = []

        for f in factures:
            num = valeur_facture(f, "invoice_no", "numero") or "?"
            vendeur = valeur_facture(f, "seller_name", "vendeur") or "?"
            items = f.get("items", [])

            articles = []

            for it in items:
                desc = str(it.get("description") or "").lower()

                if terme in desc:
                    articles.append(it.get("description"))

            if articles:
                resultats.append(
                    {
                        "num": num,
                        "vendeur": vendeur,
                        "articles": articles,
                    }
                )

        if resultats:
            res = f"Facture(s) contenant **{terme.upper()}** :\n\n"

            for r in resultats:
                res += f"• Facture n°{r['num']} — {r['vendeur']}\n"

                for art in r["articles"]:
                    res += f"   └─ {art}\n"

            return res

        return f"Aucune facture ne contient un article mentionnant **{terme.upper()}**."

    # Facture la plus ancienne
    if "ancienne" in q or "ancien" in q or "premiere" in q or "premier" in q:
        dates = []

        for f in factures:
            dt = parse_date_soigneux(f.get("date"))

            if dt:
                dates.append((dt, f))

        if dates:
            dt_min, f_anc = min(dates, key=lambda x: x[0])
            num = valeur_facture(f_anc, "invoice_no", "numero") or "?"
            vendeur = valeur_facture(f_anc, "seller_name", "vendeur") or "?"
            ttc = valeur_facture(f_anc, "gross_worth_total", "total") or "?"

            return (
                f"📅 **Facture la plus ancienne :**\n\n"
                f"• N° facture : {num}\n"
                f"• Date : {f_anc.get('date')}\n"
                f"• Vendeur : {vendeur}\n"
                f"• Total TTC : {ttc} €"
            )

    # Facture la plus récente
    if "recente" in q or "récent" in q or "derniere" in q or "dernier" in q:
        dates = []

        for f in factures:
            dt = parse_date_soigneux(f.get("date"))

            if dt:
                dates.append((dt, f))

        if dates:
            dt_max, f_rec = max(dates, key=lambda x: x[0])
            num = valeur_facture(f_rec, "invoice_no", "numero") or "?"
            vendeur = valeur_facture(f_rec, "seller_name", "vendeur") or "?"
            ttc = valeur_facture(f_rec, "gross_worth_total", "total") or "?"

            return (
                f"📅 **Facture la plus récente :**\n\n"
                f"• N° facture : {num}\n"
                f"• Date : {f_rec.get('date')}\n"
                f"• Vendeur : {vendeur}\n"
                f"• Total TTC : {ttc} €"
            )

    # Recherche par numéro
    match_num = re.search(r"\b\d{4,}\b", q)

    if match_num and not any(
        k in q for k in ["superieur", "supérieur", "dépass", "depass", "plus de"]
    ):
        target = match_num.group()

        facture = next(
            (
                f
                for f in factures
                if str(valeur_facture(f, "invoice_no", "numero") or "") == target
            ),
            None,
        )

        if facture:
            vendeur = valeur_facture(facture, "seller_name", "vendeur") or "Non renseigné"
            adresse_vendeur = facture.get("seller_address") or "Adresse non renseignée"
            client = valeur_facture(facture, "client_name", "client") or "Non renseigné"
            adresse_client = facture.get("client_address") or "Adresse non renseignée"
            total = valeur_facture(facture, "gross_worth_total", "total") or "Non renseigné"
            iban = facture.get("iban") or "Non renseigné"

            if "adresse" in q and ("vendeur" in q or "fournisseur" in q):
                return f"Adresse du vendeur **{vendeur}** pour la facture n°{target} :\n\n{adresse_vendeur}"

            if "adresse" in q and "client" in q:
                return f"Adresse du client **{client}** pour la facture n°{target} :\n\n{adresse_client}"

            if "iban" in q:
                return f"L'IBAN de la facture n°{target} est : **{iban}**"

            if "vendeur" in q or "fournisseur" in q:
                return f"Le vendeur de la facture n°{target} est : **{vendeur}**"

            if "client" in q:
                return f"Le client de la facture n°{target} est : **{client}**"

            if "montant" in q or "total" in q or "ttc" in q:
                return f"Le total TTC de la facture n°{target} est : **{total} €**"

            return (
                f"Facture n°{target}\n\n"
                f"• Vendeur : {vendeur}\n"
                f"• Client : {client}\n"
                f"• Total TTC : {total} €\n"
                f"• IBAN : {iban}"
            )

    # Nombre factures
    if "combien" in q and "facture" in q:
        return f"Il y a **{len(factures)} facture(s)** enregistrée(s)."

    # Liste vendeurs
    if ("liste" in q or "tous" in q or "quels" in q) and (
        "vendeur" in q or "fournisseur" in q
    ):
        vendeurs = sorted(
            list(
                {
                    valeur_facture(f, "seller_name", "vendeur")
                    for f in factures
                    if valeur_facture(f, "seller_name", "vendeur")
                }
            )
        )

        if not vendeurs:
            return "Aucun vendeur trouvé."

        return "🏢 **Vendeurs enregistrés :**\n\n" + "\n".join(
            f"• {v}" for v in vendeurs
        )

    # Total dépenses
    if "total" in q or "dépense" in q or "depense" in q or "somme" in q:
        total = 0.0
        count = 0

        for f in factures:
            m = _to_float(valeur_facture(f, "gross_worth_total", "total"))

            if m is not None:
                total += m
                count += 1

        return f"Le total TTC des dépenses ({count} facture(s)) est de **{total:.2f} €**."

    # Facture montant maximum
    if "plus élevé" in q or "plus eleve" in q or "maximum" in q or "plus cher" in q:
        candidats = []

        for f in factures:
            m = _to_float(valeur_facture(f, "gross_worth_total", "total"))

            if m is not None:
                candidats.append((m, f))

        if candidats:
            montant, fmax = max(candidats, key=lambda x: x[0])
            num = valeur_facture(fmax, "invoice_no", "numero") or "?"
            vendeur = valeur_facture(fmax, "seller_name", "vendeur") or "?"

            return (
                f"La facture avec le montant le plus élevé est "
                f"la facture n°{num} ({vendeur}) avec **{montant:.2f} €**."
            )

    # Fallback LLM
    if llm_status["disponible"]:
        base_clean = []

        for f in factures:
            base_clean.append(
                {
                    "no_facture": valeur_facture(f, "invoice_no", "numero"),
                    "date": f.get("date"),
                    "iban": f.get("iban"),
                    "vendeur": valeur_facture(f, "seller_name", "vendeur"),
                    "adresse_vendeur": f.get("seller_address"),
                    "client": valeur_facture(f, "client_name", "client"),
                    "adresse_client": f.get("client_address"),
                    "total_ht": f.get("net_worth_total"),
                    "total_tva": f.get("vat_total"),
                    "total_ttc": valeur_facture(f, "gross_worth_total", "total"),
                    "articles": f.get("items", []),
                }
            )

        try:
            return appeler_llm_texte(
                "Tu es un assistant d'audit bancaire. "
                "Réponds uniquement à partir des données fournies. "
                "Si l'information n'existe pas, dis-le clairement.",
                f"DONNÉES :\n{json.dumps(base_clean, ensure_ascii=False, indent=2)}\n\nQUESTION : {question}",
            )

        except Exception as e:
            return f"Erreur IA : {e}"

    return (
        "Je peux répondre à : nombre de factures, liste des vendeurs, total des dépenses, "
        "facture la plus ancienne/récente, recherche par numéro, recherche d'articles Dell/HP/PC."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🏦 FinTech Compliance")
    st.markdown(
        """
**Modules du système :**

- OCR Tesseract  
- Extraction IA du tableau  
- Auto-correction comptable  
- Audit IBAN / doublons  
- Dashboard analytics  
- Assistant sur l'historique
"""
    )

    st.divider()

    if llm_status["disponible"]:
        st.success(f"IA opérationnelle : `{llm_status['modele']}`")
    else:
        st.warning("IA locale indisponible")

    st.caption("Projet de Fin d'Études")


# ============================================================
# AFFICHAGE EXTRACTION
# ============================================================

def afficher_resultats():
    data = st.session_state.get("extraction")

    if not data:
        return

    infos = data["infos"]
    items = data["items"]
    audit = data.get("audit", {})

    st.success("Extraction terminée ✅")

    st.divider()
    st.subheader("🛡️ Audit de Conformité Bancaire")

    score = audit.get("score", 0)
    decision = audit.get("decision", "—")
    statut = audit.get("statut_long", "—")

    c1, c2 = st.columns([1, 2])

    c1.metric("Score", f"{score} / 100")
    c2.metric("Décision", decision)

    if score >= 80:
        st.success(f"**Statut :** {statut}")
    elif score >= 50:
        st.warning(f"**Statut :** {statut}")
    else:
        st.error(f"**Statut :** {statut}")

    if audit.get("alertes"):
        st.write("📌 **Points détectés :**")
        for a in audit["alertes"]:
            st.write(f"• {a}")
    else:
        st.success("Aucun risque détecté.")

    st.divider()
    st.subheader("Informations générales")

    i1, i2, i3 = st.columns(3)
    i1.metric("N° facture", infos.get("invoice_no") or "—")
    i2.metric("Date", infos.get("date") or "—")
    i3.metric("IBAN", infos.get("iban") or "—")

    st.divider()
    s_col, c_col = st.columns(2)

    with s_col:
        st.markdown("### 🏢 Vendeur")
        st.write(infos.get("seller_name") or "—")
        st.caption(infos.get("seller_address") or "")

    with c_col:
        st.markdown("### 👤 Client")
        st.write(infos.get("client_name") or "—")
        st.caption(infos.get("client_address") or "")

    st.divider()
    st.subheader("Totaux")

    t1, t2, t3 = st.columns(3)
    t1.metric("Total HT", f"{infos.get('net_worth_total') or '—'} €")
    t2.metric("TVA", f"{infos.get('vat_total') or '—'} €")
    t3.metric("Total TTC", f"{infos.get('gross_worth_total') or '—'} €")

    st.divider()
    st.subheader(f"Articles détectés ({len(items)})")

    if items:
        st.dataframe(
            items,
            width="stretch",
            hide_index=True,
            column_config={
                "statut": "Contrôle",
                "numero": "N°",
                "description": "Description",
                "qty": "Quantité",
                "um": "Unité",
                "net_price": "Prix unitaire",
                "net_worth": "Montant HT",
                "vat_percent": "TVA",
                "gross_worth": "Montant TTC",
                "alertes": "Auto-correction",
                "valide": None,
            },
        )
    else:
        st.warning("Aucun article détecté.")


# ============================================================
# INTERFACE
# ============================================================

st.title("💳 Système d'Audit Bancaire & Numérisation de Factures")
st.caption(
    "Plateforme OCR, extraction IA, auto-correction comptable et analyse financière"
)

tab_facture, tab_historique, tab_analytics, tab_assistant = st.tabs(
    [
        "📤 Traiter une facture",
        "🗂️ Historique & Export Excel",
        "📊 Dashboard Analytics",
        "💬 Assistant",
    ]
)


# ============================================================
# ONGLET 1 : EXTRACTION
# ============================================================

with tab_facture:
    fichier = st.file_uploader(
        "Choisis une facture",
        type=["png", "jpg", "jpeg"],
    )

    if fichier:
        suffixe = "." + fichier.name.rsplit(".", 1)[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffixe) as tmp:
            tmp.write(fichier.getvalue())
            chemin = tmp.name

        st.image(chemin, width="stretch")

        if st.button("🚀 Lancer l'extraction & l'audit", type="primary"):
            try:
                barre = st.progress(10, text="Analyse vendeur / client...")

                position = extraire_seller_client_par_position(chemin)
                vendeur = parser_bloc_entite(position["seller_bloc"])
                client = parser_bloc_entite(position["client_bloc"])

                barre.progress(40, text="OCR...")

                image = nettoyer_image_depuis_chemin(chemin)
                texte = extraire_texte(image)

                barre.progress(60, text="Extraction des informations...")

                infos, texte_corrige = extraire_infos_facture(texte)

                infos["seller_name"] = vendeur.get("nom")
                infos["seller_address"] = vendeur.get("adresse")
                infos["client_name"] = client.get("nom")
                infos["client_address"] = client.get("adresse")

                barre.progress(80, text="Lecture du tableau...")

                items, source, erreur = extraire_tableau_rapide(texte_corrige)

                factures_existantes = liste_factures()
                audit = auditer_facture_bancaire(infos, items, factures_existantes)

                enregistrer_facture(infos, fichier.name, texte, items)

                st.session_state.extraction = {
                    "infos": infos,
                    "items": items,
                    "texte": texte,
                    "nom_fichier": fichier.name,
                    "source_items": source,
                    "audit": audit,
                }

                barre.progress(100, text="Terminé ✅")
                barre.empty()

            except Exception as e:
                st.error(f"Erreur : {e}")

        afficher_resultats()


# ============================================================
# ONGLET 2 : HISTORIQUE
# ============================================================

with tab_historique:
    st.subheader("Historique des factures numérisées")

    factures = liste_factures()
    st.metric("Total factures en base", len(factures))

    if factures:
        df_hist = pd.DataFrame(factures)
        st.dataframe(df_hist, width="stretch", hide_index=True)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            excel = generer_excel_comptable(factures)
            st.download_button(
                label="📊 Exporter le journal comptable Excel",
                data=excel,
                file_name=f"Journal_Comptable_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with col2:
            if st.button("🗑️ Vider tout l'historique", type="secondary"):
                if vider_historique():
                    st.session_state.extraction = None

                    if "messages" in st.session_state:
                        st.session_state.messages = []

                    st.success("Historique vidé avec succès.")
                    st.rerun()

                else:
                    st.error("Erreur lors de la suppression.")

    else:
        st.info("Aucune facture enregistrée.")


# ============================================================
# ONGLET 3 : DASHBOARD
# ============================================================

with tab_analytics:
    st.subheader("📊 Tableau de Bord Financier & Analytics")
    st.caption("Visualisation interactive des dépenses extraites automatiquement.")

    factures_anal = liste_factures()

    if not factures_anal:
        st.info("Traite au moins une facture pour afficher le dashboard.")

    else:
        df = pd.DataFrame(factures_anal)

        df["ttc_num"] = df["gross_worth_total"].apply(lambda x: _to_float(x) or 0.0)
        df["ht_num"] = df["net_worth_total"].apply(lambda x: _to_float(x) or 0.0)
        df["tva_num"] = df["vat_total"].apply(lambda x: _to_float(x) or 0.0)

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            kpi_card("Volume total traité", f"{df['ttc_num'].sum():,.2f} €")

        with k2:
            kpi_card("Panier moyen", f"{df['ttc_num'].mean():,.2f} €")

        with k3:
            kpi_card("TVA cumulée", f"{df['tva_num'].sum():,.2f} €")

        with k4:
            kpi_card("Factures traitées", len(df))

        st.divider()

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("### 🍩 Répartition des dépenses par vendeur")

            vendeur_df = (
                df.groupby("seller_name")["ttc_num"]
                .sum()
                .reset_index()
                .sort_values("ttc_num", ascending=False)
            )

            vendeur_df.columns = ["Vendeur", "Total TTC (€)"]

            fig_donut = px.pie(
                vendeur_df,
                names="Vendeur",
                values="Total TTC (€)",
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )

            fig_donut.update_traces(
                textinfo="percent+label",
                marker=dict(line=dict(color="#FFFFFF", width=2)),
            )

            fig_donut.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=370,
                showlegend=True,
            )

            st.plotly_chart(fig_donut, use_container_width=True)

        with col_g2:
            st.markdown("### 📊 Top vendeurs par volume TTC")

            vendeur_sorted = vendeur_df.sort_values("Total TTC (€)", ascending=True)

            fig_bar = px.bar(
                vendeur_sorted,
                x="Total TTC (€)",
                y="Vendeur",
                orientation="h",
                color="Total TTC (€)",
                color_continuous_scale="Blues",
            )

            fig_bar.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=370,
                showlegend=False,
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        st.markdown("### 💶 Décomposition HT + TVA par facture")

        df_stack = df[["invoice_no", "ht_num", "tva_num"]].copy()
        df_stack.columns = ["N° Facture", "Montant HT (€)", "TVA (€)"]

        fig_stack = px.bar(
            df_stack,
            x="N° Facture",
            y=["Montant HT (€)", "TVA (€)"],
            color_discrete_map={
                "Montant HT (€)": "#2563eb",
                "TVA (€)": "#f59e0b",
            },
        )

        fig_stack.update_layout(
            barmode="stack",
            height=420,
            margin=dict(t=30, b=30, l=30, r=30),
        )

        st.plotly_chart(fig_stack, use_container_width=True)


# ============================================================
# ONGLET 4 : ASSISTANT
# ============================================================

with tab_assistant:
    st.subheader("Assistant factures")
    st.caption("Recherche intelligente dans les factures et leurs articles.")

    factures = liste_factures_completes()

    if not factures:
        st.info("Traite d'abord quelques factures pour utiliser l'assistant.")

    else:
        exemples = [
            "Combien de factures sont enregistrées ?",
            "Quelle est la facture la plus ancienne ?",
            "Y a-t-il des factures contenant un ordinateur Dell ?",
            "Quel est le total de toutes les dépenses ?",
        ]

        c1, c2 = st.columns(2)

        for i, exemple in enumerate(exemples):
            colonne = c1 if i % 2 == 0 else c2

            if colonne.button(exemple, key=f"q_{i}", width="stretch"):
                st.session_state["question_assistant"] = exemple

        st.divider()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        saisie = st.chat_input(
            "Ex : Y a-t-il des factures contenant un ordinateur Dell ?"
        )

        auto = st.session_state.pop("question_assistant", None)
        question = auto or saisie

        if question:
            with st.spinner("Analyse..."):
                reponse = repondre_assistant(question, factures)

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reponse,
                }
            )

            st.rerun()

        if st.session_state.messages:
            if st.button("🗑️ Effacer la conversation"):
                st.session_state.messages = []
                st.rerun()