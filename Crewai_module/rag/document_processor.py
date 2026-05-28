"""Document processor for SEC filings - chunks and prepares text for embedding."""

import re
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import html2text
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Process SEC filings for RAG system."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks (defaults to settings)
            chunk_overlap: Overlap between chunks (defaults to settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        
        logger.info(
            f"DocumentProcessor initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
    
    def process_html_file(self, file_path: Path) -> str:
        """Process HTML/XML SEC filing and convert to clean text.
        
        Args:
            file_path: Path to HTML/XML file
        
        Returns:
            Clean text content
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(content, 'lxml')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'meta', 'link']):
                element.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            text = self._clean_text(text)
            
            logger.info(f"Processed {len(text)} characters from {file_path.name}")
            return text
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and formatting.
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with single newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Additional metadata to attach to chunks
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if metadata is None:
            metadata = {}
        
        chunks = []
        start = 0
        text_length = len(text)
        chunk_id = 0
        
        while start < text_length:
            end = start + self.chunk_size
            
            # If not at the end, try to break at a sentence or paragraph
            if end < text_length:
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break != -1 and para_break > start:
                    end = para_break
                else:
                    # Look for sentence break
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break != -1 and sentence_break > start:
                        end = sentence_break + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk_data = {
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'start_char': start,
                    'end_char': end,
                    **metadata
                }
                chunks.append(chunk_data)
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < text_length else text_length
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract common sections from 10-Q filing.
        
        Args:
            text: Full filing text
        
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        # Common 10-Q sections
        section_patterns = {
            'business': r'(?:ITEM\s+1[\.:]?\s*BUSINESS|PART\s+I.*?ITEM\s+1)',
            'risk_factors': r'ITEM\s+1A[\.:]?\s*RISK\s+FACTORS',
            'md_and_a': r'ITEM\s+2[\.:]?\s*MANAGEMENT.*?DISCUSSION.*?ANALYSIS',
            'financial_statements': r'ITEM\s+1[\.:]?\s*FINANCIAL\s+STATEMENTS',
            'controls': r'ITEM\s+4[\.:]?\s*CONTROLS\s+AND\s+PROCEDURES',
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                start = match.start()
                # Find next section or end
                next_item = re.search(r'ITEM\s+\d+', text[start + 50:], re.IGNORECASE)
                if next_item:
                    end = start + 50 + next_item.start()
                else:
                    end = len(text)
                
                sections[section_name] = text[start:end].strip()
                logger.debug(f"Extracted section: {section_name}")
        
        return sections
    
    def process_and_chunk_file(
        self,
        file_path: Path,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Process file and create chunks in one step.
        
        Args:
            file_path: Path to SEC filing
            metadata: Additional metadata
        
        Returns:
            List of chunks with metadata
        """
        text = self.process_html_file(file_path)
        
        # Add file metadata
        if metadata is None:
            metadata = {}
        metadata['source_file'] = str(file_path.name)
        
        chunks = self.chunk_text(text, metadata)
        
        logger.info(
            f"Processed and chunked {file_path.name}: "
            f"{len(text)} chars -> {len(chunks)} chunks"
        )
        
        return chunks
