"""
Gmail Service - OAuth2 authentication and email sending
"""
import os
import base64
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import settings

logger = logging.getLogger(__name__)


class GmailService:
    """Handles Gmail API operations"""
    
    def __init__(self):
        self.creds = None
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate using OAuth2"""
        scopes = settings.GMAIL_SCOPES + settings.SHEETS_SCOPES
        
        if os.path.exists(settings.TOKEN_PATH):
            self.creds = Credentials.from_authorized_user_file(settings.TOKEN_PATH, scopes)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.CREDENTIALS_PATH, scopes
                )
                self.creds = flow.run_local_server(port=0)
            
            with open(settings.TOKEN_PATH, 'w') as token:
                token.write(self.creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=self.creds)
        logger.info("[Gmail] Authenticated successfully")
    
    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        ics_attachment: Optional[bytes] = None
    ) -> dict:
        """
        Send an email with optional CC and ICS attachment
        
        Returns:
            dict with 'success' (bool) and 'message_id' or 'error'
        """
        try:
            message = MIMEMultipart()
            message['From'] = settings.SENDER_EMAIL
            message['To'] = to
            message['Subject'] = subject
            
            if cc:
                message['Cc'] = cc
            
            # HTML body
            html_body = body.replace('\n', '<br>')
            message.attach(MIMEText(html_body, 'html'))
            
            # ICS attachment
            if ics_attachment:
                part = MIMEBase('text', 'calendar', method='REQUEST')
                part.set_payload(ics_attachment)
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename='meeting_invite.ics'
                )
                message.attach(part)
            
            # Encode and send
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            result = self.service.users().messages().send(
                userId='me',
                body={'raw': raw}
            ).execute()
            
            logger.info(f"[Gmail] Email sent to {to} (ID: {result['id']})")
            return {'success': True, 'message_id': result['id']}
            
        except HttpError as e:
            error_msg = f"Gmail API error: {e}"
            logger.error(f"[Gmail] {error_msg}")
            return {'success': False, 'error': error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"[Gmail] {error_msg}")
            return {'success': False, 'error': error_msg}


# Singleton instance
_gmail_service = None

def get_gmail_service() -> GmailService:
    """Get or create Gmail service singleton"""
    global _gmail_service
    if _gmail_service is None:
        _gmail_service = GmailService()
    return _gmail_service
