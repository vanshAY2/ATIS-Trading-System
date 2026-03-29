"""
ATIS v5.0 — Angel One SmartAPI Connection Test
Tests: TOTP generation → authentication → LTP fetch → profile.
"""
import sys

# Check dependencies first
for pkg in ["pyotp", "logzero"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[!] Missing: {pkg}. Run: pip install -r requirements.txt")
        sys.exit(1)

import pyotp
from SmartApi import SmartConnect

API_KEY      = "jbTc0qFF"
CLIENT_ID    = "R340624"
PASSWORD     = "5415"
TOTP_SECRET  = "3RJ7DRO6OOQNY65AXOVQRWFRBY"

print("=" * 60)
print("  Angel One SmartAPI — Connection Test")
print("=" * 60)

# Step 1: TOTP
try:
    totp = pyotp.TOTP(TOTP_SECRET).now()
    print(f"[✓] TOTP generated: {totp}")
except Exception as e:
    print(f"[✗] TOTP failed: {e}")
    sys.exit(1)

# Step 2: Init
try:
    obj = SmartConnect(api_key=API_KEY)
    print(f"[✓] SmartConnect initialized")
except Exception as e:
    print(f"[✗] SmartConnect init failed: {e}")
    sys.exit(1)

# Step 3: Login
try:
    data = obj.generateSession(CLIENT_ID, PASSWORD, totp)
    if data.get("status"):
        auth_token = data["data"]["jwtToken"]
        feed_token = obj.getfeedToken()
        refresh_token = data["data"]["refreshToken"]
        print(f"[✓] LOGIN SUCCESS")
        print(f"    Client: {CLIENT_ID}")
        print(f"    Token:  {auth_token[:30]}...")
    else:
        print(f"[✗] LOGIN FAILED: {data.get('message', data)}")
        sys.exit(1)
except Exception as e:
    print(f"[✗] Login error: {e}")
    sys.exit(1)

# Step 4: Fetch NIFTY 50 LTP
try:
    ltp_data = obj.ltpData("NSE", "NIFTY", "99926000")
    if ltp_data.get("status"):
        print(f"[✓] NIFTY 50 LTP: ₹{ltp_data['data']['ltp']}")
    else:
        print(f"[!] LTP: {ltp_data.get('message', 'Market may be closed — this is OK')}")
except Exception as e:
    print(f"[!] LTP error (normal outside market hours): {e}")

# Step 5: Profile
try:
    profile = obj.getProfile(refresh_token)
    if profile.get("status"):
        p = profile["data"]
        print(f"[✓] Profile: {p.get('name', 'N/A')} | Broker: {p.get('broker', 'N/A')}")
    else:
        print(f"[!] Profile: {profile.get('message', 'N/A')}")
except Exception as e:
    print(f"[!] Profile error: {e}")

# Cleanup
try:
    obj.terminateSession(CLIENT_ID)
    print(f"[✓] Session terminated")
except:
    pass

print("=" * 60)
print("  API KEY STATUS: VERIFIED ✓" if "auth_token" in dir() else "  API KEY STATUS: FAILED ✗")
print("=" * 60)
