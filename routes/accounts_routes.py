from fastapi import APIRouter, HTTPException
from services.account_service import get_accounts, add_account, delete_account, test_account

router = APIRouter()

@router.get("")
def list_accounts():
    return get_accounts()

@router.post("")
def create_account(data: dict):
    return add_account(data)

@router.delete("/{account_id}")
def remove_account(account_id: str):
    return delete_account(account_id)

@router.post("/{account_id}/test")
def test_connection(account_id: str):
    return test_account(account_id)
