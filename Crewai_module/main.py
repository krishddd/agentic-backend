"""Main entry point for Financial Crew."""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from crew import create_financial_crew
from config.settings import settings
from utils.logger import get_logger
from utils.validators import validate_ticker
from llm_router import print_agent_registry, check_all_models

logger = get_logger(__name__)
console = Console()


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              FINANCIAL CREW AI ANALYST                    ║
    ║         Multi-Agent Investment Analysis System            ║
    ║                                                           ║
    ║  • Research Analyst (Market & Competitive Intelligence)   ║
    ║  • Financial Analyst (SEC Filings & Metrics via RAG)      ║
    ║  • Investment Advisor (Synthesis & Recommendations)       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def main():
    """Main entry point."""
    print_banner()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Financial Crew - Multi-Agent Investment Analysis System"
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol to analyze (e.g., TSLA, AAPL)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    args = parser.parse_args()
    
    # Validate ticker
    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    
    # Override settings if needed
    if args.verbose:
        settings.verbose = True
    if args.no_cache:
        settings.cache_enabled = False
    
    # Display settings
    console.print("\n[bold cyan]Configuration:[/bold cyan]")
    console.print(f"  Ticker: [green]{ticker}[/green]")
    console.print(f"  LLM Provider: [yellow]Ollama (multi-model)[/yellow]")
    console.print(f"  Verbose: [yellow]{settings.verbose}[/yellow]")
    console.print(f"  Cache: [yellow]{'Enabled' if settings.cache_enabled else 'Disabled'}[/yellow]")
    console.print(f"  Agent Delegation: [yellow]{'Enabled' if settings.enable_delegation else 'Disabled'}[/yellow]")
    console.print()
    
    # Print agent → LLM registry
    print_agent_registry()
    
    # Check Ollama model availability
    model_status = check_all_models()
    if not model_status:
        console.print("[red]Error:[/red] Ollama is not running. Please start with: ollama serve")
        sys.exit(1)
    
    try:
        # Create crew
        console.print(f"\n[bold cyan]Initializing Financial Crew for {ticker}...[/bold cyan]\n")
        crew = create_financial_crew(ticker)
        
        # Run analysis
        console.print("[bold cyan]Starting Multi-Agent Analysis...[/bold cyan]\n")
        console.print("This may take several minutes as agents collaborate and analyze data.\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Analyzing {ticker}...", total=None)
            
            result = crew.run()
            
            progress.update(task, description=f"[green]Analysis complete for {ticker}!")
        
        # Display result
        console.print("\n" + "=" * 80 + "\n")
        console.print(Panel.fit(
            f"[bold green]Investment Report Generated Successfully![/bold green]\n\n"
            f"Report saved to: [yellow]{settings.outputs_dir / f'{ticker}_investment_report.md'}[/yellow]",
            title="Success",
            border_style="green"
        ))
        console.print("\n" + "=" * 80 + "\n")
        
        # Print preview
        console.print("[bold cyan]Report Preview:[/bold cyan]\n")
        result_str = str(result)
        preview = result_str[:1000] + "..." if len(result_str) > 1000 else result_str
        console.print(preview)
        
        console.print(f"\n\n[bold cyan]Full report available at:[/bold cyan] {settings.outputs_dir / f'{ticker}_investment_report.md'}")
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Analysis interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        console.print(f"\n[red]Error:[/red] {e}")
        console.print("\n[yellow]Please check logs for details:[/yellow]")
        console.print(f"  {settings.logs_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
