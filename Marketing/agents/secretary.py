"""
Agent Secretary - Communication Agent
Drafts and sends professional emails based on call notes
"""
from datetime import datetime
from typing import Dict, Optional

from config import settings
from services.sheets_service import get_sheets_service, SheetsService
from services.gmail_service import get_gmail_service, GmailService
from services.rag_engine import get_rag_engine, RAGEngine
from services.calendar_generator import get_calendar_generator, CalendarGenerator
from services.llm_service import get_llm_for_agent, LLMService
from services.evaluation_service import AgentEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentSecretary:
    """
    Communication Agent - The "Associate"
    
    Workflow:
    1. Find rows with Mail_Status = "Pending"
    2. Lock row (set to "Processing")
    3. Get RAG context from knowledge base
    4. Generate email via LLM
    5. Create calendar invite if scheduling mentioned
    6. Send email via Gmail
    7. Update status to "Sent" or "Failed"
    """
    
    def __init__(
        self,
        sheets: SheetsService = None,
        gmail: GmailService = None,
        rag: RAGEngine = None,
        calendar: CalendarGenerator = None,
        llm: LLMService = None
    ):
        self.sheets = sheets or get_sheets_service()
        self.gmail = gmail or get_gmail_service()
        self.rag = rag or get_rag_engine()
        self.calendar = calendar or get_calendar_generator()
        self.llm = llm or get_llm_for_agent("secretary")
        self.evaluator: Optional[AgentEvaluator] = None
    
    def run(self) -> Dict[str, int]:
        """Run the secretary agent"""
        # Initialize evaluator
        self.evaluator = AgentEvaluator(
            agent_id="secretary",
            objective="Process pending rows, generate emails, and send via Gmail"
        )
        self.evaluator.start_evaluation()
        
        stats = {'processed': 0, 'sent': 0, 'failed': 0, 'reset': 0}
        
        logger.info("=" * 60)
        logger.info("🤖 SECRETARY AGENT - Starting")
        logger.info("=" * 60)
        
        self.evaluator.record_step("agent_start", True)
        
        # Reset stuck rows
        stuck_rows = self.sheets.get_stuck_processing_rows(settings.PROCESSING_TIMEOUT_MINUTES)
        for row in stuck_rows:
            logger.warning(f"Resetting stuck row: {row['Row_ID']}")
            self.sheets.reset_to_pending(row['_row_number'])
            stats['reset'] += 1
        
        # Get pending rows
        pending_rows = self.sheets.get_pending_rows()
        logger.info(f"Found {len(pending_rows)} pending rows")
        
        if not pending_rows:
            logger.info("No pending rows to process")
            return stats
        
        for row in pending_rows:
            stats['processed'] += 1
            row_num = row['_row_number']
            
            try:
                logger.info(f"Processing Row #{row['Row_ID']}: {row['Client_Name']} @ {row['Company_Name']}")
                
                # Lock the row
                self.sheets.set_processing(row_num)
                
                # Get RAG context
                rag_context = self.rag.get_context_for_notes(
                    row['Raw_Discussion_Notes'],
                    row['Client_Requirements']
                )
                
                # Generate email
                email_content = self._generate_email(row, rag_context)
                
                # Generate calendar invite
                ics_bytes = None
                if row['Scheduling_Next_Steps'].strip():
                    ics_bytes = self.calendar.generate_ics(
                        row['Scheduling_Next_Steps'],
                        row['Client_Name'],
                        row['Company_Name'],
                        row['Call_Type']
                    )
                
                # Send email
                # Override: Send to test recipient
                test_recipient = "harish.krishna@testaing.com"
                result = self.gmail.send_email(
                    to=test_recipient,
                    subject=email_content['subject'],
                    body=email_content['body'],
                    cc=None,  # Disable CC for testing
                    ics_attachment=ics_bytes
                )
                
                if result['success']:
                    self.sheets.set_sent(row_num)
                    stats['sent'] += 1
                    logger.info(f"✓ Email sent successfully")
                else:
                    self.sheets.set_failed(row_num, result['error'])
                    stats['failed'] += 1
                    logger.error(f"✗ Email failed: {result['error']}")
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"✗ Error processing row: {error_msg}")
                self.sheets.set_failed(row_num, error_msg)
                stats['failed'] += 1
        
        logger.info(f"Completed: {stats}")
        
        # Finalize evaluation
        self.evaluator.record_step("agent_complete", True)
        evaluation = self.evaluator.finalize_evaluation()
        stats['evaluation'] = evaluation.dict()
        
        # Save evaluation to file
        import os
        eval_dir = "logs/evaluations"
        os.makedirs(eval_dir, exist_ok=True)
        eval_file = os.path.join(eval_dir, f"secretary_{evaluation.run_id}.json")
        self.evaluator.export_to_json(eval_file)
        
        return stats
    
    def _generate_email(self, row: Dict, rag_context: str) -> Dict[str, str]:
        """Generate email subject and body using LLM"""
        
        notes_lower = row['Raw_Discussion_Notes'].lower()
        urgency = "NORMAL"
        if any(word in notes_lower for word in ['urgent', 'asap', 'immediately', 'critical']):
            urgency = "URGENT"
        elif any(word in notes_lower for word in ['angry', 'frustrated', 'complaint', 'unhappy']):
            urgency = "ATTENTION REQUIRED"
        elif any(word in notes_lower for word in ['ready to buy', 'sign today', 'proceed', 'close']):
            urgency = "HOT LEAD"
        
        system_prompt = "You are a professional sales operations assistant."
        
        user_prompt = f"""Generate a professional internal email to brief the consultant.

CLIENT: {row['Client_Name']} from {row['Company_Name']}
CALL TYPE: {row['Call_Type']}
RAW NOTES: {row['Raw_Discussion_Notes']}
REQUIREMENTS: {row['Client_Requirements']}
NEXT STEPS: {row['Scheduling_Next_Steps']}
PRODUCT KNOWLEDGE: {rag_context}

Generate email with:
1. SUBJECT: [{urgency}] - Client Update: {row['Company_Name']} - {row['Call_Type']}
2. EXECUTIVE SUMMARY
3. KEY REQUIREMENTS (bullets)
4. RECOMMENDED ACTIONS
5. SENTIMENT ANALYSIS
6. ORIGINAL NOTES (at bottom)

Format:
SUBJECT: [subject]
BODY:
[body]
"""

        content = self.llm.chat(system_prompt, user_prompt)
        
        lines = content.split('\n')
        subject = ""
        body_lines = []
        in_body = False
        
        for line in lines:
            if line.startswith('SUBJECT:'):
                subject = line.replace('SUBJECT:', '').strip()
            elif line.startswith('BODY:'):
                in_body = True
            elif in_body:
                body_lines.append(line)
        
        body = '\n'.join(body_lines).strip()
        
        if not subject:
            subject = f"[{urgency}] - Client Update: {row['Company_Name']} - {row['Call_Type']}"
        if not body:
            body = content
        
        return {'subject': subject, 'body': body}


def main():
    """Run the secretary agent standalone"""
    agent = AgentSecretary()
    stats = agent.run()
    print(f"\n✓ Secretary Agent finished: {stats}")


if __name__ == "__main__":
    main()
