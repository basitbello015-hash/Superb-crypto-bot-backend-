import sqlite3
import uuid
from typing import List, Dict

DB_FILE = "accounts.db"

# -----------------------
# Initialize database
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            exchange TEXT NOT NULL,
            apiKey TEXT NOT NULL,
            secretKey TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Call on import/startup
init_db()

# -----------------------
# Account service functions
# -----------------------

def get_accounts() -> List[Dict]:
    """Retrieve all saved accounts."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, exchange, apiKey, secretKey FROM accounts")
    accounts = [dict(zip([col[0] for col in c.description], row)) for row in c.fetchall()]
    conn.close()
    return accounts

def add_account(data: dict) -> Dict:
    """Add a new account."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    account_id = str(uuid.uuid4())
    c.execute(
        "INSERT INTO accounts (id, name, exchange, apiKey, secretKey) VALUES (?, ?, ?, ?, ?)",
        (account_id, data['name'], data['exchange'], data['apiKey'], data['secretKey'])
    )
    conn.commit()
    conn.close()
    return {"status": "added", "account": {**data, "id": account_id}}

def delete_account(account_id: str) -> Dict:
    """Delete an account by ID."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "id": account_id}

def test_account(account_id: str) -> Dict:
    """Test if an account exists (mock connection test)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM accounts WHERE id = ?", (account_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": account_id, "connection": "success"}
    return {"id": account_id, "connection": "failed"}
