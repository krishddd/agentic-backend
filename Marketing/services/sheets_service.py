"""
Google Sheets Service - Read/Write operations for sales data
"""
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import settings


class SheetsService:
    """Handles Google Sheets API operations"""
    
    # Column mapping (0-indexed)
    COLUMNS = {
        'Row_ID': 0,
        'Client_Name': 1,
        'Company_Name': 2,
        'Consultant_Email': 3,
        'Manager_Email': 4,
        'Call_Type': 5,
        'Raw_Discussion_Notes': 6,
        'Client_Requirements': 7,
        'Scheduling_Next_Steps': 8,
        'Mail_Status': 9,
        'Processing_Started_At': 10,
        'Error_Log': 11,
        'Deal_Potential_Score': 12,
        'Risk_Flags': 13,
        'Strategy_Notes': 14,
        'Sync_Status': 15,
        'Nurture_Required': 16
    }
    
    def __init__(self, spreadsheet_id: str = None, sheet_name: str = None):
        self.spreadsheet_id = spreadsheet_id or settings.SPREADSHEET_ID
        self.sheet_name = sheet_name or settings.SHEET_NAME
        self.creds = None
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate using OAuth2 (shares token with Gmail)"""
        scopes = settings.GMAIL_SCOPES + settings.SHEETS_SCOPES
        
        if os.path.exists(settings.TOKEN_PATH):
            try:
                self.creds = Credentials.from_authorized_user_file(settings.TOKEN_PATH, scopes)
            except Exception as e:
                print(f"[Sheets] Error loading token: {e}")
                self.creds = None
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                    # Save refreshed token
                    with open(settings.TOKEN_PATH, 'w') as token:
                        token.write(self.creds.to_json())
                except Exception as e:
                    print(f"[Sheets] Token refresh failed: {e}")
                    print("[Sheets] Deleting invalid token and re-authenticating...")
                    # Delete the invalid token
                    if os.path.exists(settings.TOKEN_PATH):
                        os.remove(settings.TOKEN_PATH)
                    self.creds = None

            if not self.creds:
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.CREDENTIALS_PATH, scopes
                )
                self.creds = flow.run_local_server(port=0)
            
                with open(settings.TOKEN_PATH, 'w') as token:
                    token.write(self.creds.to_json())
        
        self.service = build('sheets', 'v4', credentials=self.creds)
        print("[Sheets] OK: Authenticated successfully")
    
    def get_all_rows(self) -> List[Dict[str, Any]]:
        """Get all data rows (skips header)"""
        try:
            range_name = f"{self.sheet_name}!A2:Q"
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            rows = result.get('values', [])
            return [self._row_to_dict(row, idx + 2) for idx, row in enumerate(rows)]
        except HttpError as e:
            print(f"[Sheets] ERROR: Error reading: {e}")
            return []
    
    def get_max_row_id(self) -> int:
        """Get the maximum Row_ID currently in the sheet"""
        rows = self.get_all_rows()
        if not rows:
            return 0
        ids = []
        for r in rows:
            row_id = r.get('Row_ID', '0')
            try:
                ids.append(int(row_id))
            except (ValueError, TypeError):
                continue
        return max(ids) if ids else 0
    
    def _row_to_dict(self, row: List, row_number: int) -> Dict[str, Any]:
        """Convert row list to dictionary with column names"""
        data = {'_row_number': row_number}
        for col_name, col_idx in self.COLUMNS.items():
            data[col_name] = row[col_idx] if col_idx < len(row) else ''
        return data
    
    def get_pending_rows(self) -> List[Dict[str, Any]]:
        """Get rows where Mail_Status = 'Pending'"""
        all_rows = self.get_all_rows()
        return [r for r in all_rows if r.get('Mail_Status', '').lower() == 'pending']
    
    def get_sent_waiting_rows(self) -> List[Dict[str, Any]]:
        """Get rows where Mail_Status = 'Sent' and Sync_Status = 'Waiting' or empty"""
        all_rows = self.get_all_rows()
        return [
            r for r in all_rows 
            if r.get('Mail_Status', '').lower() == 'sent' 
            and r.get('Sync_Status', '').lower() in ('waiting', '')
        ]
    
    def get_stuck_processing_rows(self, timeout_minutes: int = 10) -> List[Dict[str, Any]]:
        """Get rows stuck in 'Processing' longer than timeout"""
        all_rows = self.get_all_rows()
        stuck = []
        now = datetime.now()
        
        for row in all_rows:
            if row.get('Mail_Status', '').lower() == 'processing':
                started_at = row.get('Processing_Started_At', '')
                if started_at:
                    try:
                        start_time = datetime.fromisoformat(started_at)
                        if (now - start_time).total_seconds() > timeout_minutes * 60:
                            stuck.append(row)
                    except ValueError:
                        stuck.append(row)
                else:
                    stuck.append(row)
        
        return stuck
    
    def append_rows(self, rows: List[Dict[str, Any]]):
        """Append multiple rows to the sheet"""
        try:
            range_name = f"{self.sheet_name}!A:Q"
            values = [self._dict_to_row(row) for row in rows]
            
            self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body={'values': values}
            ).execute()
            print(f"[Sheets] OK: Appended {len(rows)} rows to sheet")
        except HttpError as e:
            print(f"[Sheets] ERROR: Error appending: {e}")
            raise

    def _dict_to_row(self, data: Dict[str, Any]) -> List[Any]:
        """Convert dictionary to row list based on COLUMNS mapping"""
        row = [''] * (max(self.COLUMNS.values()) + 1)
        for col_name, col_idx in self.COLUMNS.items():
            if col_name in data:
                row[col_idx] = data[col_name]
        return row
    
    def update_cell(self, row_number: int, column_name: str, value: str):
        """Update a single cell"""
        col_idx = self.COLUMNS.get(column_name)
        if col_idx is None:
            raise ValueError(f"Unknown column: {column_name}")
        
        col_letter = chr(ord('A') + col_idx)
        range_name = f"{self.sheet_name}!{col_letter}{row_number}"
        
        try:
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body={'values': [[value]]}
            ).execute()
            print(f"[Sheets] OK: Updated {column_name} at row {row_number}")
        except HttpError as e:
            print(f"[Sheets] ERROR: Error updating: {e}")
            raise
    
    def update_row(self, row_number: int, updates: Dict[str, str]):
        """Update multiple columns in a row"""
        for col_name, value in updates.items():
            self.update_cell(row_number, col_name, value)
    
    def set_processing(self, row_number: int):
        """Set row to Processing with timestamp"""
        self.update_row(row_number, {
            'Mail_Status': 'Processing',
            'Processing_Started_At': datetime.now().isoformat()
        })
    
    def set_sent(self, row_number: int):
        """Mark row as Sent"""
        self.update_cell(row_number, 'Mail_Status', 'Sent')
    
    def set_failed(self, row_number: int, error_message: str):
        """Mark row as Failed with error log"""
        self.update_row(row_number, {
            'Mail_Status': 'Failed',
            'Error_Log': error_message[:500]
        })
    
    def reset_to_pending(self, row_number: int):
        """Reset stuck row back to Pending"""
        self.update_row(row_number, {
            'Mail_Status': 'Pending',
            'Processing_Started_At': '',
            'Error_Log': 'Auto-reset from stuck Processing state'
        })


# Singleton instance
_sheets_service = None

def get_sheets_service(spreadsheet_id: str = None) -> SheetsService:
    """Get or create Sheets service singleton"""
    global _sheets_service
    if _sheets_service is None:
        _sheets_service = SheetsService(spreadsheet_id)
    return _sheets_service
