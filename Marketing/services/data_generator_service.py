"""
Data Generator Service - Generates synthetic sales call data using LLM
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from config import settings
from services.llm_service import get_llm_for_agent
from utils.logger import get_logger

logger = get_logger(__name__)

class DataGeneratorService:
    """Service for generating synthetic sales lead data"""
    
    def __init__(self):
        self.llm = get_llm_for_agent("data_generator")
    
    def generate_rows(self, num_rows: int = 5) -> List[Dict[str, Any]]:
        """
        Generate synthetic rows using the LLM service
        
        Returns a list of dictionaries ready for SheetsService.append_rows()
        """
        prompt = f"""Generate {num_rows} rows of realistic sales call data for a B2B CRM software company.

Create diverse scenarios including:
1. Angry client about pricing
2. Highly interested client ready to buy
3. Technical buyer asking about SSO and API
4. Client asking for discount
5. Client mentioning Salesforce (competitor)
6. Technical buyer asking about integrations
7. Urgent timeline ("need it by end of quarter")
8. Questions about data security and GDPR
9. Wanting a demo for their team
10. Budget concerns but interested

For each row provide these exact fields:
- Client_Name: Realistic name
- Company_Name: Realistic company
- Consultant_Email: firstname.lastname@company.com
- Manager_Email: sales-manager@company.com
- Call_Type: (Discovery Call, Demo, Negotiation, Support, Follow-up)
- Raw_Discussion_Notes: Messy notes (abbreviations, lowercase, incomplete sentences)
- Client_Requirements: Technical needs
- Scheduling_Next_Steps: Natural language date ("Friday 2pm", "Dec 20 at 3pm")

Output strictly as a JSON object with a key "data" containing the array of rows. No other text or markdown.
"""
        
        logger.info(f"[DataGenerator] Requesting {num_rows} synthetic rows from LLM...")
        
        try:
            response_text = self.llm.chat(
                system_prompt="You are a sales data generation assistant. Output only valid JSON.",
                user_prompt=prompt,
                json_mode=True
            )
            
            # Basic cleaning if LLM adds markdown blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            data = json.loads(response_text)
            
            # Handle different JSON structures
            rows = []
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = data.get('data', data.get('rows', [data]))
            
            if not isinstance(rows, list):
                rows = [rows]
            
            # Slice to requested number just in case
            rows = rows[:num_rows]
            
            logger.info(f"[DataGenerator] OK: Successfully generated {len(rows)} rows")
            return rows
            
        except Exception as e:
            logger.error(f"[DataGenerator] ERROR: Error generating data: {e}")
            raise RuntimeError(f"Failed to generate synthetic data: {str(e)}")

    def prepare_for_sheets(self, rows: List[Dict[str, Any]], start_id: int) -> List[Dict[str, Any]]:
        """Add metadata fields required by the sheet schema"""
        now_ts = datetime.now().isoformat()
        
        for i, row in enumerate(rows):
            row['Row_ID'] = start_id + i + 1
            row['Mail_Status'] = 'Pending'
            row['Processing_Started_At'] = ''
            row['Error_Log'] = ''
            row['Deal_Potential_Score'] = ''
            row['Risk_Flags'] = ''
            row['Strategy_Notes'] = ''
            row['Sync_Status'] = 'Waiting'
            row['Nurture_Required'] = ''
            # We can also add a generation timestamp in Strategy_Notes or elsewhere if needed
            row['Strategy_Notes'] = f"Generated via API at {now_ts}"
            
        return rows

# Singleton
_data_generator = None

def get_data_generator() -> DataGeneratorService:
    global _data_generator
    if _data_generator is None:
        _data_generator = DataGeneratorService()
    return _data_generator
