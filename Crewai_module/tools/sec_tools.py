"""Tools for searching and downloading SEC filings."""

from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
from datetime import datetime
from crewai.tools import BaseTool
from config.settings import settings
from utils.logger import get_logger
from utils.cache import cache
from utils.validators import validate_ticker, sanitize_filename

logger = get_logger(__name__)


class SECFilingSearchTool(BaseTool):
    """Tool for searching SEC filings via EDGAR API."""
    
    name: str = "SEC Filing Search"
    description: str = (
        "Search for SEC filings for a given stock ticker. "
        "Returns information about available 10-Q and 10-K filings including dates and URLs. "
        "Input should be a stock ticker symbol (e.g., 'TSLA')."
    )
    
    def _run(self, ticker: str) -> str:
        """Search for SEC filings.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Formatted string with filing information
        """
        try:
            ticker = validate_ticker(ticker)
            logger.info(f"Searching SEC filings for {ticker}")
            
            # Check cache
            cached_result = cache.get("sec_search", ticker=ticker)
            if cached_result:
                return cached_result
            
            # SEC EDGAR API endpoint
            url = f"https://data.sec.gov/submissions/CIK{self._get_cik(ticker)}.json"
            
            headers = {
                "User-Agent": settings.sec_user_agent,
                "Accept-Encoding": "gzip, deflate"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract recent filings
            filings = data.get('filings', {}).get('recent', {})
            forms = filings.get('form', [])
            filing_dates = filings.get('filingDate', [])
            accession_numbers = filings.get('accessionNumber', [])
            
            # Filter for 10-Q and 10-K
            results = []
            for i, form in enumerate(forms):
                if form in ['10-Q', '10-K'] and len(results) < 10:
                    results.append({
                        'form': form,
                        'filing_date': filing_dates[i],
                        'accession_number': accession_numbers[i]
                    })
            
            # Format output
            output = f"SEC Filings for {ticker}:\n\n"
            for filing in results:
                output += f"- {filing['form']} filed on {filing['filing_date']}\n"
                output += f"  Accession: {filing['accession_number']}\n"
            
            # Cache result
            cache.set("sec_search", output, ticker=ticker)
            
            logger.info(f"Found {len(results)} filings for {ticker}")
            return output
            
        except Exception as e:
            error_msg = f"Error searching SEC filings: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _get_cik(self, ticker: str) -> str:
        """Get CIK number for ticker (simplified - uses ticker to CIK mapping).
        
        Args:
            ticker: Stock ticker
        
        Returns:
            CIK number (padded to 10 digits)
        """
        # In production, you'd want a proper ticker->CIK mapping
        # For now, we'll use SEC's company tickers JSON
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {"User-Agent": settings.sec_user_agent}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for entry in data.values():
                if entry.get('ticker', '').upper() == ticker.upper():
                    cik = str(entry['cik_str']).zfill(10)
                    return cik
            
            raise ValueError(f"CIK not found for ticker {ticker}")
            
        except Exception as e:
            logger.error(f"Error getting CIK: {e}")
            # Return a dummy CIK (this will fail, but provides better error message)
            return "0000000000"


class SECFilingDownloadTool(BaseTool):
    """Tool for downloading SEC filings."""
    
    name: str = "SEC Filing Download"
    description: str = (
        "Download a specific SEC 10-Q or 10-K filing for a stock ticker. "
        "Input should be a ticker symbol and filing type (e.g., 'TSLA 10-Q'). "
        "Returns the file path of the downloaded filing."
    )
    
    def _run(self, input_str: str) -> str:
        """Download SEC filing.
        
        Args:
            input_str: Format: "TICKER FORM" (e.g., "TSLA 10-Q")
        
        Returns:
            Path to downloaded file or error message
        """
        try:
            parts = input_str.strip().split()
            if len(parts) < 2:
                return "Error: Input should be 'TICKER FORM' (e.g., 'TSLA 10-Q')"
            
            ticker = validate_ticker(parts[0])
            form_type = parts[1].upper()
            
            if form_type not in ['10-Q', '10-K']:
                return "Error: Form type must be '10-Q' or '10-K'"
            
            logger.info(f"Downloading {form_type} for {ticker}")
            
            # Get filing URL (simplified - in production use full EDGAR API)
            filing_url = self._get_filing_url(ticker, form_type)
            
            if not filing_url:
                return f"Could not find {form_type} filing for {ticker}"
            
            # Download filing
            headers = {"User-Agent": settings.sec_user_agent}
            response = requests.get(filing_url, headers=headers, timeout=60)
            response.raise_for_status()
            
            # Save to file
            filename = sanitize_filename(
                f"{ticker}_{form_type}_{datetime.now().strftime('%Y%m%d')}.html"
            )
            file_path = settings.filings_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded filing to {file_path}")
            return f"Successfully downloaded {form_type} for {ticker} to: {file_path}"
            
        except Exception as e:
            error_msg = f"Error downloading SEC filing: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _get_filing_url(self, ticker: str, form_type: str) -> Optional[str]:
        """Get URL for most recent filing of specified type.
        
        Args:
            ticker: Stock ticker
            form_type: Form type (10-Q or 10-K)
        
        Returns:
            URL to filing HTML or None
        """
        try:
            # Get CIK for ticker
            cik = self._get_cik(ticker)
            if not cik or cik == "0000000000":
                logger.error(f"Could not find CIK for ticker {ticker}")
                return None
            
            # Get submissions data
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            headers = {
                "User-Agent": settings.sec_user_agent,
                "Accept-Encoding": "gzip, deflate"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Find most recent filing of specified type
            filings = data.get('filings', {}).get('recent', {})
            forms = filings.get('form', [])
            accession_numbers = filings.get('accessionNumber', [])
            primary_documents = filings.get('primaryDocument', [])
            
            for i, form in enumerate(forms):
                if form == form_type:
                    accession = accession_numbers[i].replace('-', '')
                    primary_doc = primary_documents[i]
                    
                    # Construct filing URL
                    # Format: https://www.sec.gov/Archives/edgar/data/{CIK}/{ACCESSION}/{PRIMARY_DOC}
                    cik_number = cik.lstrip('0')  # Remove leading zeros
                    filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_number}/{accession}/{primary_doc}"
                    
                    logger.info(f"Found {form_type} URL for {ticker}: {filing_url}")
                    return filing_url
            
            logger.warning(f"No {form_type} filing found for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting filing URL for {ticker}: {e}")
            return None
    
    def _get_cik(self, ticker: str) -> str:
        """Get CIK number for ticker.
        
        Args:
            ticker: Stock ticker
        
        Returns:
            CIK number (padded to 10 digits)
        """
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {"User-Agent": settings.sec_user_agent}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for entry in data.values():
                if entry.get('ticker', '').upper() == ticker.upper():
                    cik = str(entry['cik_str']).zfill(10)
                    return cik
            
            raise ValueError(f"CIK not found for ticker {ticker}")
            
        except Exception as e:
            logger.error(f"Error getting CIK: {e}")
            return "0000000000"
