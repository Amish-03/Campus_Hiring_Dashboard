import os.path
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from typing import List, Optional
from ..models import EmailRecord

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailFetcher:
    def __init__(self, credentials_path: str = 'credentials.json', token_path: str = 'token.json'):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = self._get_service()

    def _get_service(self):
        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(f"Please provide '{self.credentials_path}' from Google Cloud Console.")
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        return build('gmail', 'v1', credentials=creds)

    def _extract_body(self, payload: dict) -> str:
        """Recursively extract text/plain body from nested multipart payloads."""
        mime = payload.get('mimeType', '')
        
        # Direct text/plain body
        if mime == 'text/plain' and 'data' in payload.get('body', {}):
            return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
        
        # Recurse into parts
        if 'parts' in payload:
            for part in payload['parts']:
                body = self._extract_body(part)
                if body:
                    return body
        
        return ""

    def fetch_emails(self, query: str = "from:placement_officer@kletech.ac.in after:2025/03/01", max_results: int = 200) -> List[EmailRecord]:
        all_messages = []
        results = self.service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
        all_messages.extend(results.get('messages', []))
        
        # Handle pagination
        while 'nextPageToken' in results:
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=max_results,
                pageToken=results['nextPageToken']
            ).execute()
            all_messages.extend(results.get('messages', []))
        
        fetched_emails = []
        for i, msg in enumerate(all_messages):
            email_data = self.service.users().messages().get(userId='me', id=msg['id']).execute()
            
            payload = email_data.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "No Subject")
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), "Unknown Sender")
            date = next((h['value'] for h in headers if h['name'].lower() == 'date'), "Unknown Date")
            
            # Recursive body extraction
            body = self._extract_body(payload)
            
            fetched_emails.append(EmailRecord(
                id=msg['id'],
                subject=subject,
                sender=sender,
                date=date,
                body=body,
                raw_body=str(email_data)
            ))
            safe_subject = subject[:60].encode('ascii', errors='replace').decode('ascii')
            print(f"  [{i+1}/{len(all_messages)}] {safe_subject}")
        
        return fetched_emails
