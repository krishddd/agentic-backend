"""test_mc_agreement — mc_agreement correct for majority vote."""
import os, sys, pytest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

class TestMCAgreement:
    def _calc_agreement(self, results):
        from collections import Counter
        c = Counter(results)
        total = len(results)
        if total == 0:
            return 0.0, "neutral"
        majority_sentiment, majority_count = c.most_common(1)[0]
        agreement = majority_count / total
        return agreement, majority_sentiment

    def test_unanimous_agreement(self):
        results = ["bullish", "bullish", "bullish"]
        agreement, majority = self._calc_agreement(results)
        assert agreement == 1.0
        assert majority == "bullish"

    def test_split_agreement(self):
        results = ["bullish", "bearish", "bullish"]
        agreement, majority = self._calc_agreement(results)
        assert abs(agreement - 2/3) < 0.01
        assert majority == "bullish"

    def test_three_way_split(self):
        results = ["bullish", "bearish", "neutral"]
        agreement, majority = self._calc_agreement(results)
        assert abs(agreement - 1/3) < 0.01

    def test_five_path_agreement(self):
        results = ["bearish", "bearish", "bearish", "bullish", "neutral"]
        agreement, majority = self._calc_agreement(results)
        assert abs(agreement - 3/5) < 0.01
        assert majority == "bearish"

    def test_empty_results(self):
        agreement, majority = self._calc_agreement([])
        assert agreement == 0.0
