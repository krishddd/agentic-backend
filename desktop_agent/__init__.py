"""
Desktop Agent Module — AI-powered desktop automation.

Usage:
    from desktop_agent import DesktopAgent
    agent = DesktopAgent()
    result = agent.execute("list all Python files in the current directory")

    from desktop_agent import ReceiptProcessor
    rp = ReceiptProcessor()
    result = rp.process("desktop_agent/receipts")
"""

from .agent import DesktopAgent, DesktopResult, TOOL_REGISTRY
from .receipt_processor import ReceiptProcessor
