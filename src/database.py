import sqlite3
import os

DB_NAME = "factures.db"


def initialiser_base():
    """Crée les tables SQLite si elles n'existent pas."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS factures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_no TEXT,
        date TEXT,
        seller_name TEXT,
        seller_address TEXT,
        client_name TEXT,
        client_address TEXT,
        seller_tax_id TEXT,
        client_tax_id TEXT,
        iban TEXT,
        net_worth_total TEXT,
        vat_total TEXT,
        gross_worth_total TEXT,
        fichier_image TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        facture_id INTEGER,
        numero INTEGER,
        description TEXT,
        qty TEXT,
        um TEXT,
        net_price TEXT,
        net_worth TEXT,
        vat_percent TEXT,
        gross_worth TEXT,
        FOREIGN KEY (facture_id) REFERENCES factures(id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()


def enregistrer_facture(infos, nom_fichier, texte, items):
    """
    Enregistre une facture. 
    ANTI-DOUBLON : Si le N° de facture existe déjà, supprime l'ancienne version avant d'insérer.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    invoice_no = infos.get("invoice_no")

    # --- MÉCANISME ANTI-DOUBLON ---
    if invoice_no and str(invoice_no).strip() not in ("—", "", "None", "null"):
        inv_str = str(invoice_no).strip()
        cursor.execute("SELECT id FROM factures WHERE invoice_no = ?", (inv_str,))
        anciens = cursor.fetchall()
        for row in anciens:
            old_id = row[0]
            cursor.execute("DELETE FROM items WHERE facture_id = ?", (old_id,))
            cursor.execute("DELETE FROM factures WHERE id = ?", (old_id,))

    # Insertion de la nouvelle facture
    cursor.execute("""
    INSERT INTO factures (
        invoice_no, date, seller_name, seller_address, client_name, client_address,
        seller_tax_id, client_tax_id, iban, net_worth_total, vat_total, gross_worth_total, fichier_image
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        infos.get("invoice_no"), infos.get("date"), infos.get("seller_name"), infos.get("seller_address"),
        infos.get("client_name"), infos.get("client_address"), infos.get("seller_tax_id"), infos.get("client_tax_id"),
        infos.get("iban"), infos.get("net_worth_total"), infos.get("vat_total"), infos.get("gross_worth_total"),
        nom_fichier
    ))

    facture_id = cursor.lastrowid

    # Insertion des articles
    for item in items:
        cursor.execute("""
        INSERT INTO items (
            facture_id, numero, description, qty, um, net_price, net_worth, vat_percent, gross_worth
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            facture_id, item.get("numero"), item.get("description"), item.get("qty"), item.get("um"),
            item.get("net_price"), item.get("net_worth"), item.get("vat_percent"), item.get("gross_worth")
        ))

    conn.commit()
    conn.close()


def recuperer_toutes_factures():
    """Récupère toutes les factures rangées par ordre récent."""
    if not os.path.exists(DB_NAME):
        initialiser_base()
        return []
    
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM factures ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]