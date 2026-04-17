"""Create Bybit demo account + API keys using main account credentials.

Credentials MUST be provided via environment variables (never committed):
    BYBIT_MAIN_API_KEY    — main account API key
    BYBIT_MAIN_API_SECRET — main account API secret

Example:
    export BYBIT_MAIN_API_KEY=...
    export BYBIT_MAIN_API_SECRET=...
    python scripts/create_demo_keys.py
"""

import hashlib
import hmac
import json
import os
import sys
import time

import requests

API_KEY = os.environ.get("BYBIT_MAIN_API_KEY")
API_SECRET = os.environ.get("BYBIT_MAIN_API_SECRET")
if not API_KEY or not API_SECRET:
    sys.exit(
        "ERROR: BYBIT_MAIN_API_KEY and BYBIT_MAIN_API_SECRET must be set in the "
        "environment. Never commit them to source. See .env.example for the "
        "expected variables; copy to .env and fill in your real credentials."
    )
BASE = "https://api.bybit.com"
RECV_WINDOW = "5000"


def sign(timestamp, params_str):
    sign_str = f"{timestamp}{API_KEY}{RECV_WINDOW}{params_str}"
    return hmac.new(API_SECRET.encode(), sign_str.encode(), hashlib.sha256).hexdigest()


def post(path, body):
    timestamp = str(int(time.time() * 1000))
    body_str = json.dumps(body)
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": sign(timestamp, body_str),
        "X-BAPI-RECV-WINDOW": RECV_WINDOW,
        "Content-Type": "application/json",
    }
    resp = requests.post(f"{BASE}{path}", headers=headers, json=body)
    return resp.json()


# Step 1: Create demo member
print("Step 1: Creating demo member...")
r1 = post("/v5/user/create-demo-member", {})
print(f"  Response: {json.dumps(r1, indent=2)}")

if r1.get("retCode") == 0:
    demo_uid = r1["result"].get("uid")
    print(f"  Demo UID: {demo_uid}")
elif r1.get("retCode") == 10020100:
    print("  Demo member already exists, proceeding...")
    # Get demo UID from account info
    demo_uid = None
else:
    print(f"  Error: {r1.get('retMsg')}")
    demo_uid = None

# Step 2: Create API key for demo account
print("\nStep 2: Creating demo API key...")
body2 = {
    "subuid": int(demo_uid) if demo_uid else 0,
    "note": "paper-trader",
    "readOnly": 0,  # 0 = read-write
    "permissions": {
        "ContractTrade": ["Order", "Position"],
        "Spot": [],
        "Wallet": ["AccountTransfer"],
        "Options": [],
        "Exchange": [],
    }
}

# If we don't have demo_uid, try without it
if not demo_uid:
    # Try to get sub accounts to find demo UID
    print("  Attempting to find demo UID...")
    timestamp = str(int(time.time() * 1000))
    params = "memberType=6"  # 6 = demo member type
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": sign(timestamp, params),
        "X-BAPI-RECV-WINDOW": RECV_WINDOW,
    }
    resp = requests.get(f"{BASE}/v5/user/query-sub-members?{params}", headers=headers)
    r_sub = resp.json()
    print(f"  Sub members: {json.dumps(r_sub, indent=2)}")

    if r_sub.get("retCode") == 0:
        members = r_sub.get("result", {}).get("subMembers", [])
        for m in members:
            if "demo" in m.get("username", "").lower() or m.get("memberType") == 6:
                demo_uid = m["uid"]
                print(f"  Found demo UID: {demo_uid}")
                break

if demo_uid:
    body2["subuid"] = int(demo_uid)
    r2 = post("/v5/user/create-subuid-apikey", body2)
    print(f"  Response: {json.dumps(r2, indent=2)}")

    if r2.get("retCode") == 0:
        result = r2["result"]
        demo_key = result.get("apiKey", "")
        demo_secret = result.get("secret", "")
        print(f"\n{'='*60}")
        print(f"  DEMO API KEY:    {demo_key}")
        print(f"  DEMO API SECRET: {demo_secret}")
        print(f"{'='*60}")
        print(f"\n  Add to .env:")
        print(f"  BYBIT_API_KEY={demo_key}")
        print(f"  BYBIT_API_SECRET={demo_secret}")
    else:
        print(f"  Error: {r2.get('retMsg')}")
else:
    print("  Could not determine demo UID. Try creating demo member first.")
