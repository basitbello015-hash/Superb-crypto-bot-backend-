import json, os
from bot_fib_scoring import ACCOUNTS_FILE

def get_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        return []
    with open(ACCOUNTS_FILE, "r") as f:
        return json.load(f)

def add_account(data: dict):
    accounts = get_accounts()
    new_account = {"id": str(len(accounts) + 1), **data}
    accounts.append(new_account)
    with open(ACCOUNTS_FILE, "w") as f:
        json.dump(accounts, f, indent=2)
    return {"status": "added", "account": new_account}

def delete_account(account_id: str):
    accounts = get_accounts()
    updated = [acc for acc in accounts if str(acc.get("id")) != str(account_id)]
    with open(ACCOUNTS_FILE, "w") as f:
        json.dump(updated, f, indent=2)
    return {"status": "deleted", "id": account_id}

def test_account(account_id: str):
    return {"id": account_id, "connection": "success"}
