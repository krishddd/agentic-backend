"""
Agent Analyst - Strategic Intelligence Agent
Scores deals, detects risks, and routes alerts
"""
import json
import re
from typing import Dict, List

import requests

from config import settings
from services.sheets_service import get_sheets_service, SheetsService
from services.llm_service import get_llm_for_agent, LLMService
from services.evaluation_service import AgentEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentAnalyst:
    """
    Intelligence Agent - The "Strategist"
    
    Workflow:
    1. Find rows with Mail_Status = "Sent" and Sync_Status = "Waiting"
    2. Analyze for deal potential (score 1-10)
    3. Detect competitor mentions
    4. Extract risk flags
    5. Route alerts (Slack for high scores, email fallback)
    6. Update sheet with analysis
    """
    
    def __init__(self, sheets: SheetsService = None, llm: LLMService = None):
        self.sheets = sheets or get_sheets_service()
        self.llm = llm or get_llm_for_agent("analyst")
        self.competitors = self._load_competitors()
        self.evaluator: Optional[AgentEvaluator] = None
    
    def _load_competitors(self) -> Dict:
        """Load competitor battlecard data"""
        try:
            with open(settings.COMPETITORS_PATH, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} competitor battlecards")
                return data
        except FileNotFoundError:
            logger.warning("Competitors file not found")
            return {}
    
    def run(self) -> Dict[str, int]:
        """Run the analyst agent"""
        # Initialize evaluator
        self.evaluator = AgentEvaluator(
            agent_id="analyst",
            objective="Analyze sent emails, score deals, and detect risks"
        )
        self.evaluator.start_evaluation()
        
        stats = {'analyzed': 0, 'high_score': 0, 'low_score': 0, 'alerts_sent': 0}
        
        logger.info("=" * 60)
        logger.info("ANALYST AGENT - Starting")
        logger.info("=" * 60)
        
        self.evaluator.record_step("agent_start", True)
        
        rows = self.sheets.get_sent_waiting_rows()
        logger.info(f"Found {len(rows)} rows to analyze")
        
        if not rows:
            logger.info("No rows waiting for analysis")
            return stats
        
        for row in rows:
            stats['analyzed'] += 1
            row_num = row['_row_number']
            
            try:
                logger.info(f"Analyzing Row #{row['Row_ID']}: {row['Company_Name']}")
                
                competitors_found = self._detect_competitors(row['Raw_Discussion_Notes'])
                analysis = self._analyze_deal(row, competitors_found)
                
                score = analysis['deal_score']
                if score >= 8:
                    stats['high_score'] += 1
                    if self._send_slack_alert(row, analysis):
                        stats['alerts_sent'] += 1
                elif score <= 5:
                    stats['low_score'] += 1
                    analysis['nurture_required'] = 'Yes'
                else:
                    analysis['nurture_required'] = ''
                
                self.sheets.update_row(row_num, {
                    'Deal_Potential_Score': str(score),
                    'Risk_Flags': analysis['risk_flags'],
                    'Strategy_Notes': analysis['strategy_notes'],
                    'Sync_Status': 'Synced',
                    'Nurture_Required': analysis.get('nurture_required', '')
                })
                
                logger.info(f"Score: {score}/10, Risks: {analysis['risk_flags'][:50]}...")
                
            except Exception as e:
                logger.error(f"Error analyzing row: {e}")
                self.sheets.update_cell(row_num, 'Sync_Status', 'Error')
        
        logger.info(f"Completed: {stats}")
        
        # Finalize evaluation
        self.evaluator.record_step("agent_complete", True)
        evaluation = self.evaluator.finalize_evaluation()
        stats['evaluation'] = evaluation.dict()
        
        # Save evaluation to file
        import os
        eval_dir = "logs/evaluations"
        os.makedirs(eval_dir, exist_ok=True)
        eval_file = os.path.join(eval_dir, f"analyst_{evaluation.run_id}.json")
        self.evaluator.export_to_json(eval_file)
        
        return stats
    
    def _detect_competitors(self, notes: str) -> List[Dict]:
        """Detect competitor mentions"""
        found = []
        notes_lower = notes.lower()
        
        for comp_name, comp_data in self.competitors.items():
            if comp_name.lower() in notes_lower:
                found.append({
                    'name': comp_name,
                    'kill_points': comp_data.get('kill_points', []),
                    'counter_objections': comp_data.get('counter_objections', {})
                })
        
        return found
    
    def _analyze_deal(self, row: Dict, competitors: List[Dict]) -> Dict:
        """Use LLM to analyze deal potential"""
        
        competitor_context = ""
        if competitors:
            for c in competitors:
                competitor_context += f"\n- {c['name']} mentioned. Kill points: {c['kill_points'][:2]}"
        
        system_prompt = "You are a strategic sales analyst. Provide precise analysis."
        
        user_prompt = f"""Analyze this sales call:

CLIENT: {row['Client_Name']} from {row['Company_Name']}
CALL TYPE: {row['Call_Type']}
NOTES: {row['Raw_Discussion_Notes']}
REQUIREMENTS: {row['Client_Requirements']}
COMPETITORS: {competitor_context if competitor_context else "None mentioned"}

Provide EXACTLY in this format:
DEAL_SCORE: [1-10]
RISK_FLAGS: [comma-separated concerns]
STRATEGY_NOTES: [2-3 sentences]
SENTIMENT: [Positive/Neutral/Negative]
"""

        content = self.llm.chat(system_prompt, user_prompt, temperature=0.5)
        
        result = {
            'deal_score': 5,
            'risk_flags': '',
            'strategy_notes': '',
            'sentiment': 'Neutral'
        }
        
        for line in content.split('\n'):
            if line.startswith('DEAL_SCORE:'):
                try:
                    score_str = re.search(r'\d+', line)
                    if score_str:
                        result['deal_score'] = min(10, max(1, int(score_str.group())))
                except ValueError:
                    pass
            elif line.startswith('RISK_FLAGS:'):
                result['risk_flags'] = line.replace('RISK_FLAGS:', '').strip()
            elif line.startswith('STRATEGY_NOTES:'):
                result['strategy_notes'] = line.replace('STRATEGY_NOTES:', '').strip()
            elif line.startswith('SENTIMENT:'):
                result['sentiment'] = line.replace('SENTIMENT:', '').strip()
        
        if competitors:
            comp_suggestions = [f"vs {c['name']}: {c['kill_points'][0]}" for c in competitors if c['kill_points']]
            if comp_suggestions:
                result['strategy_notes'] += f" | COMPETITIVE: {'; '.join(comp_suggestions)}"
        
        return result
    
    def _send_slack_alert(self, row: Dict, analysis: Dict) -> bool:
        """Send alert to Slack or fallback to email"""
        
        if not settings.SLACK_ENABLED:
            logger.info("Slack not configured, using email fallback")
            return self._send_email_alert(row, analysis)
        
        message = {
            "text": f"Hot Lead: {row['Company_Name']} - Score {analysis['deal_score']}/10"
        }
        
        try:
            response = requests.post(settings.SLACK_WEBHOOK_URL, json=message, timeout=10)
            if response.status_code == 200:
                logger.info(f"Slack alert sent for {row['Company_Name']}")
                return True
            else:
                return self._send_email_alert(row, analysis)
        except Exception as e:
            logger.error(f"Slack failed: {e}, using email fallback")
            return self._send_email_alert(row, analysis)
    
    def _send_email_alert(self, row: Dict, analysis: Dict) -> bool:
        """Fallback: Send alert via email"""
        from services.gmail_service import get_gmail_service
        
        gmail = get_gmail_service()
        
        subject = f"HOT LEAD: {row['Company_Name']} - Score {analysis['deal_score']}/10"
        body = f"""
<h2>Hot Lead Detected!</h2>
<p><strong>Company:</strong> {row['Company_Name']}</p>
<p><strong>Client:</strong> {row['Client_Name']}</p>
<p><strong>Deal Score:</strong> {analysis['deal_score']}/10</p>
<p><strong>Strategy:</strong> {analysis['strategy_notes']}</p>
<p><strong>Risk Flags:</strong> {analysis['risk_flags']}</p>
        """
        
        result = gmail.send_email(to=row['Manager_Email'], subject=subject, body=body)
        return result['success']


def main():
    """Run the analyst agent standalone"""
    agent = AgentAnalyst()
    stats = agent.run()
    print(f"\nAnalyst Agent finished: {stats}")


if __name__ == "__main__":
    main()
