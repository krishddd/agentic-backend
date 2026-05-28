"""Main crew orchestrator for Financial Crew multi-agent system."""

import time
from pathlib import Path
from crewai import Crew, Process
from agents import (
    create_research_analyst,
    create_financial_analyst,
    create_investment_advisor
)
from tasks import (
    create_research_task,
    create_financial_analysis_task,
    create_filing_analysis_task,
    create_report_task
)
from rag import DocumentProcessor, VectorStoreManager
from config.settings import settings
from utils.logger import get_logger
from utils.validators import validate_ticker
from evaluation.evaluation_service import AgentEvaluator
# llm_router module removed — print_agent_registry was imported but never used

logger = get_logger(__name__)


class FinancialCrew:
    """Orchestrates the financial analysis crew."""
    
    def __init__(self, stock_symbol: str):
        """Initialize the financial crew.
        
        Args:
            stock_symbol: Stock ticker symbol to analyze
        """
        self.stock_symbol = validate_ticker(stock_symbol)
        self.agents = {}
        self.tasks = {}
        self.crew = None
        self.evaluator = None
        
        logger.info(f"Initializing Financial Crew for {self.stock_symbol}")
        
        # Initialize evaluator if enabled
        if settings.enable_evaluation:
            self.evaluator = AgentEvaluator(
                agent_id="financial-crew",
                objective=f"Analyze {self.stock_symbol} stock and generate comprehensive investment report"
            )
            logger.info(f"Evaluation enabled for run {self.evaluator.run_id}")
        
        # Create agents
        self._create_agents()
        
        # Create tasks
        self._create_tasks()
        
        # Create crew
        self._create_crew()
    
    def _create_agents(self):
        """Create all agents with per-agent Ollama LLMs."""
        logger.info("Creating agents with per-agent LLM assignments...")
        
        self.agents['research'] = create_research_analyst()
        self.agents['financial'] = create_financial_analyst()
        self.agents['investment'] = create_investment_advisor()
        
        # Log per-agent LLM assignments
        for name, agent in self.agents.items():
            model_name = getattr(agent.llm, 'model', 'unknown')
            logger.info(f"  Agent '{name}' → LLM: {model_name}")
        
        logger.info(f"Created {len(self.agents)} agents with multi-LLM assignment")
    
    def _create_tasks(self):
        """Create all tasks."""
        logger.info("Creating tasks...")
        
        # Task 1: Research
        self.tasks['research'] = create_research_task(
            agent=self.agents['research'],
            stock_symbol=self.stock_symbol
        )
        
        # Task 2: Financial Analysis
        self.tasks['financial_analysis'] = create_financial_analysis_task(
            agent=self.agents['financial'],
            stock_symbol=self.stock_symbol
        )
        
        # Task 3: Filing Analysis (with RAG)
        self.tasks['filing_analysis'] = create_filing_analysis_task(
            agent=self.agents['financial'],
            stock_symbol=self.stock_symbol
        )
        
        # Task 4: Report Generation (synthesizes all previous tasks)
        self.tasks['report'] = create_report_task(
            agent=self.agents['investment'],
            stock_symbol=self.stock_symbol,
            context_tasks=[
                self.tasks['research'],
                self.tasks['financial_analysis'],
                self.tasks['filing_analysis']
            ]
        )
        
        logger.info(f"Created {len(self.tasks)} tasks")
    
    def _create_crew(self):
        """Create the crew with all agents and tasks."""
        logger.info("Creating crew...")
        
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            process=Process.sequential,  # Tasks run sequentially
            verbose=settings.verbose,
            memory=True,  # Enable memory for context sharing
        )
        
        logger.info("Crew created successfully")
    
    def run(self) -> str:
        """Execute the crew and return the final report.
        
        Returns:
            Final investment report
        """
        logger.info(f"Starting analysis for {self.stock_symbol}")
        
        # Start evaluation tracking
        if self.evaluator:
            self.evaluator.start_evaluation()
        
        step_start = time.time()
        
        try:
            # Step 1: Process SEC filings
            self._track_step("Process SEC filings", lambda: self._process_sec_filings_if_available())
            
            # Step 2: Execute research task
            logger.info("Executing crew tasks...")
            
            # Track crew execution
            result = self._execute_with_tracking()
            
            # Step 3: Save results
            self._track_step("Save investment report", lambda: self._save_result(result))
            
            # Finalize and save evaluation
            if self.evaluator:
                evaluation = self.evaluator.finalize_evaluation()
                self._save_evaluation(evaluation)
                logger.info(f"Evaluation: {evaluation.overall_verdict} (score: {evaluation.overall_score:.2f})")
            
            logger.info(f"Analysis completed for {self.stock_symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error executing crew: {e}")
            if self.evaluator:
                self.evaluator.record_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    recovered=False
                )
            raise
    
    def _process_sec_filings_if_available(self):
        """Process any downloaded SEC filings into the RAG system."""
        try:
            filings_dir = settings.filings_dir
            
            # Look for filings for this ticker
            filing_files = list(filings_dir.glob(f"{self.stock_symbol}_*.html"))
            
            if not filing_files:
                logger.info(f"No pre-downloaded SEC filings found for {self.stock_symbol}")
                return
            
            logger.info(f"Found {len(filing_files)} SEC filings to process")
            
            # Initialize RAG components
            processor = DocumentProcessor()
            vector_store = VectorStoreManager()
            
            for filing_file in filing_files:
                logger.info(f"Processing {filing_file.name}")
                
                # Extract filing date from filename if possible
                # Format: TICKER_10-Q_YYYYMMDD.html
                parts = filing_file.stem.split('_')
                filing_date = parts[-1] if len(parts) >= 3 else None
                
                # Process and chunk
                chunks = processor.process_and_chunk_file(
                    filing_file,
                    metadata={
                        'ticker': self.stock_symbol,
                        'filing_type': parts[1] if len(parts) >= 2 else 'unknown'
                    }
                )
                
                # Add to vector store
                vector_store.add_documents(
                    chunks,
                    ticker=self.stock_symbol,
                    filing_date=filing_date
                )
            
            logger.info("SEC filings processed and embedded successfully")
            
        except Exception as e:
            logger.warning(f"Error processing SEC filings: {e}")
            # Don't fail the entire process if RAG processing fails
    
    def _save_result(self, result):
        """Save the final result to file.
        
        Args:
            result: Crew execution result
        """
        try:
            output_file = settings.outputs_dir / f"{self.stock_symbol}_investment_report.md"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(str(result))
            
            logger.info(f"Report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving result: {e}")
    
    def _execute_with_tracking(self) -> str:
        """Execute crew with evaluation tracking.

        Records each task step AFTER execution so success/duration are accurate.

        Returns:
            Crew execution result
        """
        task_names = ["Research Analysis", "Financial Analysis", "SEC Filing Analysis", "Report Generation"]

        # Execute the crew — capture start time for step duration
        import time as _time
        t0 = _time.time()
        error_occurred = False
        result = None
        try:
            result = self.crew.kickoff(
                inputs={'stock_symbol': self.stock_symbol}
            )
        except Exception as e:
            error_occurred = True
            raise
        finally:
            total_duration_ms = (_time.time() - t0) * 1000
            per_step_ms = total_duration_ms / len(task_names)

            # Record steps AFTER execution so success reflects actual outcome
            for task_name in task_names:
                if self.evaluator:
                    self.evaluator.record_step(
                        step_name=task_name,
                        success=not error_occurred,
                        duration_ms=round(per_step_ms, 1),
                        error_message=None if not error_occurred else "Crew execution failed",
                    )

        return result
    
    def _track_step(self, step_name: str, step_func):
        """Track a single step execution.
        
        Args:
            step_name: Name of the step
            step_func: Function to execute
        """
        start_time = time.time()
        success = False
        error_message = None
        
        try:
            step_func()
            success = True
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            if self.evaluator:
                self.evaluator.record_step(
                    step_name=step_name,
                    success=success,
                    duration_ms=duration_ms,
                    error_message=error_message
                )
    
    def _save_evaluation(self, evaluation):
        """Save the evaluation result to file.
        
        Args:
            evaluation: AgentEvaluation object
        """
        try:
            eval_dir = settings.outputs_dir / "evaluations"
            self.evaluator.save_evaluation(evaluation, eval_dir)
            logger.info(f"Evaluation saved to {eval_dir}")
        except Exception as e:
            logger.error(f"Error saving evaluation: {e}")


def create_financial_crew(stock_symbol: str) -> FinancialCrew:
    """Factory function to create a Financial Crew.
    
    Args:
        stock_symbol: Stock ticker symbol
    
    Returns:
        Configured FinancialCrew instance
    """
    return FinancialCrew(stock_symbol)
