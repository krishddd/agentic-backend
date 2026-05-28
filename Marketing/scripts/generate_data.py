"""
Data Generator - Creates synthetic sales call data
"""
import csv
import os
import sys
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_data(num_rows: int = 10) -> List[Dict]:
    """Generate synthetic sales call data using LLM"""
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
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

For each row provide:
- Client_Name: Realistic name
- Company_Name: Realistic company
- Consultant_Email: firstname.lastname@company.com
- Manager_Email: sales-manager@company.com
- Call_Type: (Discovery Call, Demo, Negotiation, Support, Follow-up)
- Raw_Discussion_Notes: Messy notes (abbreviations, lowercase, incomplete sentences)
- Client_Requirements: Technical needs
- Scheduling_Next_Steps: Natural language date ("Friday 2pm", "Dec 20 at 3pm")

Output as valid JSON array."""

    logger.info("Generating synthetic data with LLM...")
    
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": "Output only valid JSON arrays, no markdown."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        response_format={"type": "json_object"}
    )
    
    import json
    content = response.choices[0].message.content
    
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            data = data.get('data', data.get('rows', [data]))
        if not isinstance(data, list):
            data = [data]
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return []
    
    # Add default columns
    for idx, row in enumerate(data):
        row['Row_ID'] = idx + 1
        row['Mail_Status'] = 'Pending'
        row['Processing_Started_At'] = ''
        row['Error_Log'] = ''
        row['Deal_Potential_Score'] = ''
        row['Risk_Flags'] = ''
        row['Strategy_Notes'] = ''
        row['Sync_Status'] = 'Waiting'
        row['Nurture_Required'] = ''
    
    logger.info(f"Generated {len(data)} rows")
    return data


def save_to_csv(data: List[Dict], filepath: str):
    """Save data to CSV file"""
    if not data:
        logger.error("No data to save")
        return
    
    columns = [
        'Row_ID', 'Client_Name', 'Company_Name', 'Consultant_Email', 'Manager_Email',
        'Call_Type', 'Raw_Discussion_Notes', 'Client_Requirements', 'Scheduling_Next_Steps',
        'Mail_Status', 'Processing_Started_At', 'Error_Log', 'Deal_Potential_Score',
        'Risk_Flags', 'Strategy_Notes', 'Sync_Status', 'Nurture_Required'
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)
    
    logger.info(f"Saved to {filepath}")


def main():
    """Generate and save synthetic data"""
    print("=" * 50)
    print("Sales Data Generator")
    print("=" * 50)
    
    data = generate_synthetic_data(10)
    
    if not data:
        print("Failed to generate data")
        return
    
    output_path = os.path.join(settings.DATA_DIR, "synthetic_sales_data.csv")
    save_to_csv(data, output_path)
    
    print(f"\n✓ Generated {len(data)} rows")
    print(f"✓ Saved to: {output_path}")
    print(f"\nCopy this CSV to your Google Sheet named '{settings.SHEET_NAME}'")


if __name__ == "__main__":
    main()
