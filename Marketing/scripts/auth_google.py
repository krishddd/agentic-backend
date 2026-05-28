"""
Quick OAuth Authentication Script
Run this to generate a fresh token.json with all required scopes
"""
import os
from google_auth_oauthlib.flow import InstalledAppFlow

# Define all required scopes
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/spreadsheets"
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Parent dir
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")

print("=" * 50)
print("Google OAuth Authentication")
print("=" * 50)
print(f"\nScopes being requested:")
for scope in SCOPES:
    print(f"  - {scope}")
print(f"\nCredentials file: {CREDENTIALS_PATH}")
print(f"Token will be saved to: {TOKEN_PATH}")

# Delete existing token if exists
if os.path.exists(TOKEN_PATH):
    os.remove(TOKEN_PATH)
    print("\n[Deleted existing token.json]")

print("\n[Starting OAuth flow - a browser window will open...]")

try:
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    creds = flow.run_local_server(port=0)
    
    with open(TOKEN_PATH, 'w') as token:
        token.write(creds.to_json())
    
    print("\n✓ SUCCESS! token.json has been created.")
    print("You can now run the API server.")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nPossible solutions:")
    print("1. Make sure credentials.json is valid")
    print("2. Enable Gmail API and Google Sheets API in Google Cloud Console")
    print("3. Add required scopes to OAuth consent screen")
