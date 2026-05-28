"""
Scheduler - Main entry point for the Sales Operations AI Agent
"""
import time
import argparse
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schedule

from config import settings
from agents.secretary import AgentSecretary
from agents.analyst import AgentAnalyst
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def run_all_agents():
    """Run both agents in sequence"""
    logger.info("=" * 70)
    logger.info(f"⏰ SCHEDULED RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # Run Secretary first
    try:
        secretary = AgentSecretary()
        secretary_stats = secretary.run()
    except Exception as e:
        logger.error(f"Secretary agent error: {e}")
        secretary_stats = {'error': str(e)}
    
    # Then run Analyst
    try:
        analyst = AgentAnalyst()
        analyst_stats = analyst.run()
    except Exception as e:
        logger.error(f"Analyst agent error: {e}")
        analyst_stats = {'error': str(e)}
    
    logger.info("-" * 70)
    logger.info("📊 RUN SUMMARY:")
    logger.info(f"   Secretary: {secretary_stats}")
    logger.info(f"   Analyst:   {analyst_stats}")
    logger.info("-" * 70)
    
    return {'secretary': secretary_stats, 'analyst': analyst_stats}


def run_continuous():
    """Run agents on a schedule"""
    interval_minutes = settings.POLLING_INTERVAL_SECONDS // 60
    
    print("🚀 SALES OPERATIONS AI AGENT")
    print("=" * 50)
    print(f"Mode: Continuous")
    print(f"Polling: Every {interval_minutes} minutes")
    print(f"Email: {settings.SENDER_EMAIL}")
    print(f"Sheet ID: {settings.SPREADSHEET_ID or 'Not configured'}")
    print("=" * 50)
    
    if not settings.SPREADSHEET_ID:
        print("\n⚠️  SPREADSHEET_ID not set in config/settings.py")
    
    run_all_agents()
    
    schedule.every(settings.POLLING_INTERVAL_SECONDS).seconds.do(run_all_agents)
    
    print(f"\n⏳ Next run in {interval_minutes} minutes (Ctrl+C to stop)\n")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")


def run_once():
    """Run agents once"""
    print("🚀 SALES OPERATIONS AI AGENT")
    print("=" * 50)
    print("Mode: One-shot")
    print("=" * 50)
    
    results = run_all_agents()
    print("\n✅ Complete")
    return results


def main():
    """Main entry point"""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Sales Operations AI Agent")
    parser.add_argument('--mode', choices=['once', 'continuous'], default='once')
    parser.add_argument('--secretary-only', action='store_true')
    parser.add_argument('--analyst-only', action='store_true')
    
    args = parser.parse_args()
    
    if args.secretary_only:
        AgentSecretary().run()
        return
    
    if args.analyst_only:
        AgentAnalyst().run()
        return
    
    if args.mode == 'continuous':
        run_continuous()
    else:
        run_once()


if __name__ == "__main__":
    main()
