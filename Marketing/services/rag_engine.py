"""
RAG Engine - Knowledge Base retrieval for email drafting
"""
import re
from typing import List

from config import settings


class RAGEngine:
    """Retrieval Augmented Generation for product knowledge"""
    
    def __init__(self):
        self.knowledge_base = ""
        self.sections = {}
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load and parse knowledge base file"""
        try:
            with open(settings.KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
                self.knowledge_base = f.read()
            self._parse_sections()
            print(f"[RAG] ✓ Loaded knowledge base ({len(self.knowledge_base)} chars)")
        except FileNotFoundError:
            print(f"[RAG] ✗ Knowledge base not found at {settings.KNOWLEDGE_BASE_PATH}")
            self.knowledge_base = ""
    
    def _parse_sections(self):
        """Parse knowledge base into sections by headers"""
        current_section = "general"
        current_content = []
        
        for line in self.knowledge_base.split('\n'):
            if line.startswith('## '):
                if current_content:
                    self.sections[current_section] = '\n'.join(current_content)
                current_section = line[3:].strip().lower()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            self.sections[current_section] = '\n'.join(current_content)
    
    def extract_questions(self, text: str) -> List[str]:
        """Extract potential questions from raw notes"""
        questions = []
        
        question_patterns = [
            r'(?:asked about|wanted to know|inquired about|curious about)\s+([^,.;]+)',
            r'(?:does it|do you|can we|is there|how much|what is|when)\s+([^?]+)\??',
            r'(?:need|require|looking for)\s+([^,.;]+)',
        ]
        
        text_lower = text.lower()
        for pattern in question_patterns:
            matches = re.findall(pattern, text_lower)
            questions.extend([m.strip() for m in matches])
        
        tech_terms = ['sso', 'api', 'saml', 'security', 'pricing', 'trial', 'integration', 
                      'migrate', 'mobile', 'uptime', 'support', 'contract', 'gdpr']
        for term in tech_terms:
            if term in text_lower:
                questions.append(term)
        
        return list(set(questions))
    
    def retrieve(self, query: str, max_context: int = 1000) -> str:
        """Retrieve relevant context from knowledge base"""
        query_lower = query.lower()
        relevant_parts = []
        
        section_mapping = {
            'price': 'pricing tiers',
            'cost': 'pricing tiers',
            'sso': 'technical specifications',
            'saml': 'technical specifications',
            'api': 'technical specifications',
            'security': 'technical specifications',
            'integration': 'technical specifications',
            'trial': 'frequently asked questions',
            'contract': 'frequently asked questions',
            'migrate': 'frequently asked questions',
            'mobile': 'frequently asked questions',
        }
        
        matched_sections = set()
        for term, section in section_mapping.items():
            if term in query_lower:
                matched_sections.add(section)
        
        for section_name, content in self.sections.items():
            if section_name in matched_sections or any(term in section_name for term in query_lower.split()):
                relevant_parts.append(content)
        
        if not relevant_parts:
            for line in self.knowledge_base.split('\n'):
                if any(word in line.lower() for word in query_lower.split()):
                    relevant_parts.append(line)
        
        result = '\n'.join(relevant_parts)[:max_context]
        return result if result else "No specific information found in knowledge base."
    
    def get_context_for_notes(self, raw_notes: str, requirements: str = "") -> str:
        """Get all relevant context for drafting an email"""
        combined_text = f"{raw_notes} {requirements}"
        questions = self.extract_questions(combined_text)
        
        if not questions:
            return "No specific technical questions identified."
        
        context_parts = []
        for q in questions[:5]:
            ctx = self.retrieve(q, max_context=500)
            if ctx and ctx != "No specific information found in knowledge base.":
                context_parts.append(f"**{q.upper()}**:\n{ctx}")
        
        if not context_parts:
            return "No specific product information needed for this call."
        
        return "\n\n".join(context_parts)


# Singleton instance
_rag_engine = None

def get_rag_engine() -> RAGEngine:
    """Get or create RAG engine singleton"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
