"""
Desktop Agent Module — LLM-routed desktop automation.

Wires existing desktop_tools (FileSystemExecutor, CodeInterpreter, PDFTools,
SystemMonitor, BrowserTools, ScreenshotTools, etc.) into an LLM-driven agent
that can execute natural language desktop tasks.

Architecture:
    User prompt -> LLM classifies tool + params -> Execute tool -> Return result

Integrated with the Multi-Agent Orchestrator via FastAPI endpoint /desktop/execute.
"""

import os
import sys
import json
import time
import logging
import re
import requests
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # Multi_Agentic_Testaing_V1/
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

# ─── LLM Config ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DESKTOP_LLM = os.getenv("DESKTOP_LLM_MODEL", "qwen3:8b")

# ─── Import Desktop Tools ───────────────────────────────────────────────────
try:
    from src.epistemic_agent.desktop_tools import (
        FileSystemExecutor, FileResult,
        CodeInterpreter,
        SystemMonitor, SystemInfo, ProcessInfo,
        PDFTools, PDFResult,
        BrowserTools, BrowserResult,
        ScreenshotTools, ScreenshotResult,
        ImageTools, ImageResult,
        ClipboardTools,
    )
    TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[DesktopAgent] Desktop tools import failed: {e}")
    TOOLS_AVAILABLE = False


# ===========================================================================
# Tool Registry — maps tool names to descriptions for LLM classification
# ===========================================================================
TOOL_REGISTRY = {
    "file_list": {
        "description": "List files in a directory, optionally with pattern matching",
        "params": ["path", "pattern", "recursive"],
        "examples": ["list files in Downloads", "show all .py files", "what's in this folder"],
    },
    "file_read": {
        "description": "Read the contents of a text file",
        "params": ["path"],
        "examples": ["read config.yaml", "show the contents of README.md", "open the log file"],
    },
    "file_write": {
        "description": "Write or append content to a file",
        "params": ["path", "content", "mode"],
        "examples": ["create a file called notes.txt", "write 'hello' to output.txt", "save this data"],
    },
    "file_search": {
        "description": "Search for files by name or by content inside files",
        "params": ["path", "query", "content_search"],
        "examples": ["find all PDF files", "search for files containing 'TODO'", "find the config file"],
    },
    "file_info": {
        "description": "Get detailed metadata about a file (size, dates, permissions)",
        "params": ["path"],
        "examples": ["file info for report.pdf", "how big is this file", "when was it modified"],
    },
    "file_delete": {
        "description": "Delete a file (moves to trash by default)",
        "params": ["path"],
        "examples": ["delete temp.txt", "remove the old backup"],
    },
    "file_copy": {
        "description": "Copy a file or directory to a new location",
        "params": ["src", "dst"],
        "examples": ["copy report.md to backup/", "duplicate this file"],
    },
    "file_move": {
        "description": "Move or rename a file",
        "params": ["src", "dst"],
        "examples": ["rename output.txt to results.txt", "move file to archive/"],
    },
    "dir_tree": {
        "description": "Show directory tree structure",
        "params": ["path", "max_depth"],
        "examples": ["show directory tree", "project structure", "folder layout"],
    },
    "dir_create": {
        "description": "Create a new directory",
        "params": ["path"],
        "examples": ["create folder called output", "make a new directory"],
    },
    "code_execute": {
        "description": "Execute Python or Shell code in a sandboxed environment",
        "params": ["code", "language"],
        "examples": ["run this python script", "execute 'pip list'", "calculate 2**100", "run shell command"],
    },
    "system_info": {
        "description": "Get system information (CPU, RAM, disk, OS, Python version)",
        "params": [],
        "examples": ["show system info", "how much RAM is available", "what OS is this"],
    },
    "system_processes": {
        "description": "List running processes sorted by memory usage",
        "params": ["top_n"],
        "examples": ["show running processes", "what's using the most memory", "top processes"],
    },
    "system_packages": {
        "description": "List installed Python packages",
        "params": [],
        "examples": ["list installed packages", "is pandas installed", "show pip packages"],
    },
    "pdf_create": {
        "description": "Create a PDF document from text or markdown content",
        "params": ["content", "output_path", "title"],
        "examples": ["create a PDF report", "convert this to PDF", "generate PDF document"],
    },
    "pdf_read": {
        "description": "Extract text from an existing PDF file",
        "params": ["path"],
        "examples": ["read this PDF", "extract text from report.pdf", "what does the PDF say"],
    },
    "pdf_merge": {
        "description": "Merge multiple PDF files into one",
        "params": ["file_list", "output_path"],
        "examples": ["merge these PDFs", "combine report1.pdf and report2.pdf"],
    },
    "pdf_info": {
        "description": "Get metadata and page count of a PDF",
        "params": ["path"],
        "examples": ["PDF info", "how many pages in this PDF"],
    },
    "screenshot": {
        "description": "Take a screenshot of the desktop or a specific region",
        "params": ["region", "save_path"],
        "examples": ["take a screenshot", "capture the screen", "screenshot this"],
    },
    "browser_open": {
        "description": "Open a URL in a browser and extract page content",
        "params": ["url"],
        "examples": ["open google.com", "visit this website", "browse to the URL"],
    },
    "browser_search": {
        "description": "Search Google and return structured results",
        "params": ["query", "num_results"],
        "examples": ["search Google for AI agents", "google this topic"],
    },
    "receipt_scan": {
        "description": "Scan a folder for receipt images and return inventory",
        "params": ["folder"],
        "examples": ["scan receipts folder", "how many receipts are there", "inventory receipts"],
    },
    "receipt_process": {
        "description": "Process receipt images: OCR extract data, classify categories, organize into folders, generate Excel report",
        "params": ["folder", "org_method", "category_logic"],
        "examples": [
            "process all receipts", "organize and categorize receipts", "run receipt pipeline",
            "analyze and arrange the files in desktop_agent/receipts",
            "process files in this folder and arrange them",
            "scan receipts and create expense report",
        ],
    },
    "receipt_report": {
        "description": "Generate Excel spreadsheet summary from processed receipts",
        "params": ["folder", "output_path"],
        "examples": ["generate receipt report", "create expense spreadsheet", "summarize receipts in Excel"],
    },
}


# ===========================================================================
# LLM Helper — call Ollama for tool classification
# ===========================================================================
def _call_ollama(prompt: str, system: str = "", model: str = DESKTOP_LLM,
                 timeout: int = 60, max_tokens: int = 2048) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json={
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        }, timeout=timeout)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        # Strip <think>...</think> blocks from qwen3
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        logger.error(f"[DesktopAgent] Ollama call failed: {e}")
        return ""


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM output."""
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    # Try code block
    match = re.search(r'```(?:json)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except:
            pass
    # Try raw braces
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {}


# ===========================================================================
# Desktop Agent — LLM-routed tool executor
# ===========================================================================
@dataclass
class DesktopResult:
    """Result from a desktop agent execution."""
    success: bool
    tool: str
    action: str
    result: Any = None
    error: str = ""
    duration_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure result is JSON-serializable
        if hasattr(d["result"], "__dict__"):
            d["result"] = str(d["result"])
        return d


class DesktopAgent:
    """
    LLM-powered desktop automation agent.

    Routes natural language tasks to the appropriate desktop tool,
    executes them, and returns structured results.

    Uses existing desktop_tools from src/epistemic_agent/desktop_tools/.
    """

    def __init__(self, base_path: str = "."):
        if not TOOLS_AVAILABLE:
            raise RuntimeError("Desktop tools not available. Check imports.")

        self.fs = FileSystemExecutor(base_path=base_path)
        self.code = CodeInterpreter()
        self.system = SystemMonitor()
        self.pdf = PDFTools()
        self.clipboard = ClipboardTools()
        self.execution_log: List[Dict] = []
        logger.info("[DesktopAgent] Initialized with all desktop tools")

    # ── Tool Classification ──────────────────────────────────────────────

    def classify_tool(self, task: str) -> Dict[str, Any]:
        """Use LLM to classify which tool to use and extract parameters."""
        tool_descriptions = "\n".join(
            f"- {name}: {info['description']} (params: {info['params']})"
            for name, info in TOOL_REGISTRY.items()
        )

        prompt = f"""Classify this desktop task into the correct tool and extract parameters.

AVAILABLE TOOLS:
{tool_descriptions}

USER TASK: {task}

Respond with ONLY a JSON object:
{{
  "tool": "tool_name",
  "params": {{"param1": "value1", ...}},
  "explanation": "brief explanation of what will be done"
}}

Rules:
- Use file paths relative to current directory unless absolute path given
- For code_execute, put the code in the "code" param
- For file_write, put the content in the "content" param
- If the task is ambiguous, pick the most likely tool
"""

        response = _call_ollama(
            prompt,
            system="You are a desktop automation tool router. Classify user tasks into "
                   "the correct tool and extract parameters. Respond with JSON only. No thinking or explanation outside JSON.",
            max_tokens=1024,
        )

        classification = _extract_json(response)
        if not classification.get("tool"):
            # Keyword fallback
            classification = self._keyword_fallback(task)

        return classification

    def _keyword_fallback(self, task: str) -> Dict[str, Any]:
        """Fallback tool classification using keywords."""
        task_lower = task.lower()

        # Receipt-related keywords must come BEFORE generic ones to avoid
        # "process" matching system_processes when user means receipt_process.
        keyword_map = [
            # Receipt processing — checked first to avoid false matches
            (["receipt report", "expense report", "receipt excel", "receipt spreadsheet"], "receipt_report"),
            (["scan receipt", "inventory receipt"], "receipt_scan"),
            (["receipt", "receipts", "expense", "ocr receipt",
              "analyze and process", "analyse and process",
              "arrange files", "arrange in proper", "arrange in folder",
              "categorize files", "categorise files",
              "organize images", "organise images",
              "sort receipt", "process images",
              "process the folder", "process folder files",
              "process files in"], "receipt_process"),
            # File operations
            (["list files", "show files", "what files", "directory listing", "ls ", "dir "], "file_list"),
            (["read file", "show contents", "open file", "cat "], "file_read"),
            (["write file", "create file", "save to"], "file_write"),
            (["search file", "find file", "locate"], "file_search"),
            (["file info", "file size", "file details"], "file_info"),
            (["delete file", "remove file", "rm "], "file_delete"),
            (["copy file", "duplicate"], "file_copy"),
            (["move file", "rename file", "mv "], "file_move"),
            (["directory tree", "folder structure", "tree"], "dir_tree"),
            (["create folder", "make directory", "mkdir"], "dir_create"),
            # Code execution
            (["run code", "execute code", "execute python", "python script",
              "script", "calculate", "compute", "shell", "pip "], "code_execute"),
            # System
            (["system info", "cpu", "ram", "memory", "disk", "os info"], "system_info"),
            (["process list", "running process", "top process", "ps "], "system_processes"),
            (["packages", "installed", "pip list"], "system_packages"),
            # PDF
            (["create pdf", "generate pdf", "make pdf"], "pdf_create"),
            (["read pdf", "extract pdf", "pdf text"], "pdf_read"),
            (["merge pdf", "combine pdf"], "pdf_merge"),
            (["pdf info", "pdf pages", "pdf metadata"], "pdf_info"),
            # Browser
            (["screenshot", "capture screen"], "screenshot"),
            (["open url", "browse", "visit", "website"], "browser_open"),
            (["google", "search web"], "browser_search"),
        ]

        for keywords, tool in keyword_map:
            if any(kw in task_lower for kw in keywords):
                extracted_params = {"raw_task": task}
                # For receipt tools, extract folder path from prompt
                if tool in ("receipt_process", "receipt_scan", "receipt_report"):
                    extracted_params["folder"] = self._extract_folder_path(task)
                return {"tool": tool, "params": extracted_params,
                        "explanation": f"Matched keyword -> {tool}"}

        return {"tool": "code_execute", "params": {"code": f"print('Task: {task}')"}, "explanation": "Default fallback"}

    @staticmethod
    def _extract_folder_path(task: str) -> str:
        """Extract a folder path from a natural language prompt.

        Examples:
          'process files in desktop_agent/receipts' -> 'desktop_agent/receipts'
          'analyze C:/Users/data/receipts folder'   -> 'C:/Users/data/receipts'
          'organize all receipts'                   -> 'desktop_agent/receipts' (default)
        """
        import re
        # Try to find a path-like string (contains / or \)
        # Match: word/word or C:\path\to or /absolute/path
        m = re.search(r'([A-Za-z]:[\\/][\w\\/.\-]+|[\w./\-]+/[\w./\-]+)', task)
        if m:
            path = m.group(1).strip().rstrip('/\\')
            # Clean up trailing words that aren't part of path
            path = re.sub(r'\s+(folder|directory|path|and|in|the)$', '', path, flags=re.I)
            return path
        return "desktop_agent/receipts"  # default

    # ── Tool Execution ───────────────────────────────────────────────────

    def execute(self, task: str) -> DesktopResult:
        """Execute a desktop task end-to-end: classify -> execute -> return."""
        start = time.time()

        # 1. Classify
        classification = self.classify_tool(task)
        tool_name = classification.get("tool", "unknown")
        params = classification.get("params", {})
        explanation = classification.get("explanation", "")

        logger.info(f"[DesktopAgent] Task: {task[:60]}...")
        logger.info(f"[DesktopAgent] Tool: {tool_name} | Params: {json.dumps(params, default=str)[:100]}")

        # 2. Execute
        try:
            result = self._dispatch(tool_name, params, task)
            duration = int((time.time() - start) * 1000)
            desktop_result = DesktopResult(
                success=True, tool=tool_name,
                action=explanation, result=result,
                duration_ms=duration,
            )
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            logger.error(f"[DesktopAgent] Execution error: {e}")
            desktop_result = DesktopResult(
                success=False, tool=tool_name,
                action=explanation, error=str(e),
                duration_ms=duration,
            )

        # 3. Log
        self.execution_log.append(desktop_result.to_dict())
        return desktop_result

    def _dispatch(self, tool: str, params: dict, raw_task: str = "") -> Any:
        """Dispatch to the correct tool executor."""

        # ── File Operations ──────────────────────────────────────────
        if tool == "file_list":
            r = self.fs.list_files(
                path=params.get("path", "."),
                pattern=params.get("pattern", "*"),
                recursive=params.get("recursive", False),
            )
            return str(r) if hasattr(r, 'message') else r

        elif tool == "file_read":
            r = self.fs.read_file(filepath=params.get("path", params.get("filepath", "")))
            return {"message": r.message, "data": r.data} if r.success else {"error": r.error}

        elif tool == "file_write":
            r = self.fs.write_file(
                filepath=params.get("path", params.get("filepath", "output.txt")),
                content=params.get("content", ""),
                mode=params.get("mode", "w"),
            )
            return str(r)

        elif tool == "file_search":
            r = self.fs.search_files(
                path=params.get("path", "."),
                query=params.get("query", params.get("raw_task", "*")),
                content_search=params.get("content_search", False),
            )
            return str(r) if hasattr(r, 'message') else r

        elif tool == "file_info":
            r = self.fs.get_file_info(filepath=params.get("path", params.get("filepath", "")))
            return {"message": r.message, "data": r.data} if r.success else {"error": r.error}

        elif tool == "file_delete":
            r = self.fs.delete_file(filepath=params.get("path", ""))
            return str(r)

        elif tool == "file_copy":
            r = self.fs.copy_file(src=params.get("src", ""), dst=params.get("dst", ""))
            return str(r)

        elif tool == "file_move":
            r = self.fs.move_file(src=params.get("src", ""), dst=params.get("dst", ""))
            return str(r)

        elif tool == "dir_tree":
            r = self.fs.get_directory_tree(
                path=params.get("path", "."),
                max_depth=params.get("max_depth", 3),
            )
            return str(r) if hasattr(r, 'message') else r

        elif tool == "dir_create":
            r = self.fs.create_directory(path=params.get("path", ""))
            return str(r)

        # ── Code Execution ───────────────────────────────────────────
        elif tool == "code_execute":
            code = params.get("code", params.get("raw_task", ""))
            language = params.get("language", None)
            r = self.code.execute_with_retry(code=code, language=language)
            return {
                "status": r.status.value,
                "stdout": r.stdout or "",
                "stderr": r.stderr or "",
                "return_value": str(r.return_value) if r.return_value else None,
                "duration": r.duration_seconds,
            }

        # ── System Monitor ───────────────────────────────────────────
        elif tool == "system_info":
            info = self.system.get_system_info()
            return str(info)

        elif tool == "system_processes":
            top_n = params.get("top_n", 15)
            procs = self.system.get_running_processes(top_n=int(top_n))
            return [str(p) for p in procs]

        elif tool == "system_packages":
            pkgs = self.system.get_installed_packages()
            return pkgs[:50]  # Limit output

        # ── PDF Tools ────────────────────────────────────────────────
        elif tool == "pdf_create":
            r = self.pdf.create_pdf(
                content=params.get("content", ""),
                output_path=params.get("output_path", "output.pdf"),
                title=params.get("title", "Document"),
            )
            return {"success": r.success, "message": r.message, "path": r.path}

        elif tool == "pdf_read":
            r = self.pdf.read_pdf(filepath=params.get("path", params.get("filepath", "")))
            return {"success": r.success, "message": r.message, "data": r.data}

        elif tool == "pdf_merge":
            file_list = params.get("file_list", [])
            output = params.get("output_path", "merged.pdf")
            r = self.pdf.merge_pdfs(file_list=file_list, output_path=output)
            return {"success": r.success, "message": r.message, "path": r.path}

        elif tool == "pdf_info":
            r = self.pdf.pdf_info(filepath=params.get("path", params.get("filepath", "")))
            return {"success": r.success, "message": r.message, "data": r.data}

        # ── Screenshot ───────────────────────────────────────────────
        elif tool == "screenshot":
            try:
                ss = ScreenshotTools()
                r = ss.capture_screen(
                    save_path=params.get("save_path", f"screenshot_{int(time.time())}.png"),
                )
                return {"success": r.success, "message": r.message, "path": r.path}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── Browser ──────────────────────────────────────────────────
        elif tool == "browser_open":
            try:
                browser = BrowserTools(headless=True)
                r = browser.open_url(url=params.get("url", ""))
                browser.close()
                return {"success": r.success, "message": r.message, "data": r.data}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif tool == "browser_search":
            try:
                browser = BrowserTools(headless=True)
                r = browser.search_google(
                    query=params.get("query", params.get("raw_task", "")),
                    num_results=params.get("num_results", 5),
                )
                browser.close()
                return {"success": r.success, "message": r.message, "data": r.data}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── Receipt Processing ─────────────────────────────────────────
        elif tool in ("receipt_scan", "receipt_process", "receipt_report"):
            try:
                from desktop_agent.receipt_processor import ReceiptProcessor
                rp = ReceiptProcessor(base_path=str(self.fs._safe_path(".")))
                folder = params.get("folder", "desktop_agent/receipts")

                if tool == "receipt_scan":
                    return rp.scan_folder(folder)
                elif tool == "receipt_process":
                    config = {
                        "org_method": params.get("org_method", "move"),
                        "category_logic": params.get("category_logic", "auto_detect"),
                    }
                    return rp.process(folder, config=config)
                elif tool == "receipt_report":
                    scan = rp.scan_folder(folder)
                    if scan.get("error"):
                        return scan
                    rp.receipts = rp.ocr_batch(scan["images"])
                    rp.receipts = rp.classify_receipts(rp.receipts)
                    output = params.get("output_path", None)
                    path = rp.generate_report(rp.receipts, output_path=output)
                    return {"success": True, "report_path": path, "receipts": len(rp.receipts)}
            except ImportError as e:
                return {"error": f"Receipt processor not available: {e}"}
            except Exception as e:
                return {"error": f"Receipt processing failed: {e}"}

        else:
            raise ValueError(f"Unknown tool: {tool}")

    # ── Utility ──────────────────────────────────────────────────────────

    def get_capabilities(self) -> Dict[str, Any]:
        """Return all available capabilities."""
        return {
            "tools": list(TOOL_REGISTRY.keys()),
            "tool_count": len(TOOL_REGISTRY),
            "descriptions": {k: v["description"] for k, v in TOOL_REGISTRY.items()},
            "llm_model": DESKTOP_LLM,
            "ollama_url": OLLAMA_BASE_URL,
        }

    def get_execution_log(self) -> List[Dict]:
        return list(self.execution_log)
