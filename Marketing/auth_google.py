"""
Google OAuth Authentication Script — Manual Token Exchange

Avoids CSRF state mismatch by using a fixed redirect URI and 
disabling the state parameter verification.

Usage:
    cd Marketing
    python auth_google.py
"""

import os
import sys
import json
import hashlib
import secrets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import settings


def main():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    scopes = list(set(settings.GMAIL_SCOPES + settings.SHEETS_SCOPES))
    token_path = settings.TOKEN_PATH
    creds_path = settings.CREDENTIALS_PATH

    print("=" * 60)
    print("  Google OAuth Authentication")
    print("=" * 60)
    print(f"  Sender Email:   {settings.SENDER_EMAIL}")
    print(f"  Spreadsheet:    {settings.SPREADSHEET_ID}")
    print(f"  Token path:     {token_path}")
    print("=" * 60)

    if not os.path.exists(creds_path):
        print(f"\n[ERROR] credentials.json not found at: {creds_path}")
        return

    # Delete old token
    if os.path.exists(token_path):
        os.remove(token_path)
        print("[INFO] Removed old token.json")

    # ── Manual OAuth flow to avoid CSRF state mismatch ──
    # Read client config
    with open(creds_path, 'r') as f:
        client_config = json.load(f)

    # Extract client info
    if 'installed' in client_config:
        cinfo = client_config['installed']
    elif 'web' in client_config:
        cinfo = client_config['web']
    else:
        print("[ERROR] Invalid credentials.json format")
        return

    client_id = cinfo['client_id']
    client_secret = cinfo['client_secret']
    token_uri = cinfo.get('token_uri', 'https://oauth2.googleapis.com/token')
    redirect_uri = "http://localhost:8090"

    # Build auth URL (no state parameter)
    import urllib.parse
    params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': ' '.join(scopes),
        'access_type': 'offline',
        'prompt': 'consent',
    }
    auth_url = f"https://accounts.google.com/o/oauth2/auth?{urllib.parse.urlencode(params)}"

    # Start local server to capture the auth code
    import threading
    import http.server
    
    auth_code = [None]
    
    class AuthHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.urlparse(self.path).query
            parsed = urllib.parse.parse_qs(query)
            if 'code' in parsed:
                auth_code[0] = parsed['code'][0]
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<h1>Authentication complete!</h1><p>You may close this tab.</p>")
            else:
                self.send_response(400)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                error = parsed.get('error', ['unknown'])[0]
                self.wfile.write(f"<h1>Error: {error}</h1>".encode())
        
        def log_message(self, format, *args):
            pass  # Suppress server logs

    server = http.server.HTTPServer(('localhost', 8090), AuthHandler)
    server.timeout = 120  # 2 minute timeout

    # Open browser
    import webbrowser
    print(f"\n[INFO] Opening browser for authentication...")
    print(f"  Sign in with: {settings.SENDER_EMAIL}")
    webbrowser.open(auth_url)

    # Wait for callback
    print("[INFO] Waiting for authorization (2 min timeout)...\n")
    while auth_code[0] is None:
        server.handle_request()
    
    server.server_close()

    if not auth_code[0]:
        print("[ERROR] No auth code received")
        return

    print("[OK] Authorization code received!")

    # Exchange auth code for tokens
    import requests
    token_response = requests.post(token_uri, data={
        'code': auth_code[0],
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code',
    })

    if token_response.status_code != 200:
        print(f"[ERROR] Token exchange failed: {token_response.text}")
        return

    token_data = token_response.json()

    # Build credentials object and save
    creds = Credentials(
        token=token_data['access_token'],
        refresh_token=token_data.get('refresh_token'),
        token_uri=token_uri,
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes,
    )

    with open(token_path, 'w') as f:
        f.write(creds.to_json())
    print(f"[OK] Token saved to: {token_path}")

    # ── Verify connections ──
    print("\n[INFO] Testing Gmail API...")
    try:
        from googleapiclient.discovery import build
        svc = build('gmail', 'v1', credentials=creds)
        labels = svc.users().labels().list(userId='me').execute().get('labels', [])
        print(f"[OK] Gmail connected! {len(labels)} labels found.")
    except Exception as e:
        print(f"[WARN] Gmail test failed: {e}")

    print("[INFO] Testing Sheets API...")
    try:
        from googleapiclient.discovery import build
        svc = build('sheets', 'v4', credentials=creds)
        result = svc.spreadsheets().get(spreadsheetId=settings.SPREADSHEET_ID).execute()
        title = result.get('properties', {}).get('title', '?')
        tabs = [s['properties']['title'] for s in result.get('sheets', [])]
        print(f"[OK] Sheet: '{title}' | Tabs: {', '.join(tabs)}")
    except Exception as e:
        print(f"[WARN] Sheets test failed: {e}")

    print(f"\n{'='*60}")
    print(f"  DONE! Token saved. Ready to use.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
