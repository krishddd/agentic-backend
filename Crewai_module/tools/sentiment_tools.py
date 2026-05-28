"""Sentiment analysis tools for market sentiment."""

from typing import List, Dict
from crewai.tools import BaseTool
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalysisTool(BaseTool):
    """Tool for analyzing market sentiment from text."""
    
    name: str = "Sentiment Analysis"
    description: str = (
        "Analyze sentiment of text (news, articles, social media) about a stock. "
        "Input should be text to analyze for sentiment. "
        "Returns sentiment score (positive, negative, neutral) and confidence."
    )
    
    def _run(self, text: str) -> str:
        """Analyze sentiment of text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment analysis results
        """
        try:
            logger.info(f"Analyzing sentiment for text of length {len(text)}")
            
            # Simple keyword-based sentiment analysis
            # In production, use a proper NLP model like FinBERT
            sentiment_score = self._calculate_sentiment(text)
            
            if sentiment_score > 0.2:
                sentiment = "POSITIVE"
                confidence = min(sentiment_score * 100, 100)
            elif sentiment_score < -0.2:
                sentiment = "NEGATIVE"
                confidence = min(abs(sentiment_score) * 100, 100)
            else:
                sentiment = "NEUTRAL"
                confidence = 100 - abs(sentiment_score) * 100
            
            output = f"Sentiment Analysis Results:\n\n"
            output += f"Overall Sentiment: {sentiment}\n"
            output += f"Confidence: {confidence:.1f}%\n"
            output += f"Sentiment Score: {sentiment_score:.3f} (range: -1 to +1)\n\n"
            
            # Add interpretation
            if sentiment == "POSITIVE":
                output += "Interpretation: The text expresses predominantly positive sentiment, "
                output += "indicating favorable market perception or optimistic outlook.\n"
            elif sentiment == "NEGATIVE":
                output += "Interpretation: The text expresses predominantly negative sentiment, "
                output += "indicating concerns or pessimistic outlook.\n"
            else:
                output += "Interpretation: The text expresses balanced or neutral sentiment, "
                output += "without strong positive or negative indicators.\n"
            
            return output
            
        except Exception as e:
            error_msg = f"Error analyzing sentiment: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate simple sentiment score using keyword matching.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment score between -1 and 1
        """
        text_lower = text.lower()
        
        # Positive financial keywords
        positive_words = [
            'profit', 'growth', 'gain', 'increase', 'beat', 'exceed', 'strong',
            'surge', 'rise', 'boost', 'improve', 'outperform', 'bullish', 'positive',
            'success', 'record', 'high', 'upgrade', 'buy', 'opportunity', 'expansion',
            'momentum', 'breakthrough', 'win', 'revenue', 'earnings', 'innovation'
        ]
        
        # Negative financial keywords
        negative_words = [
            'loss', 'decline', 'decrease', 'fall', 'miss', 'weak', 'drop', 'cut',
            'concern', 'risk', 'bearish', 'negative', 'fail', 'lawsuit', 'investigation',
            'downgrade', 'sell', 'crash', 'plunge', 'warning', 'debt', 'deficit',
            'bankruptcy', 'recession', 'crisis', 'struggle', 'challenges'
        ]
        
        # Count occurrences
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        # Calculate score
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
        
        score = (positive_count - negative_count) / total_count
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, score))
