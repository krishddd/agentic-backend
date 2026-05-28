"""LLM-as-a-Judge validation service for ensuring judge reliability."""

import statistics
from typing import List, Dict, Any
import time

from utils.logger import get_logger

logger = get_logger(__name__)


class JudgeValidator:
    """Validates LLM-as-a-Judge reliability and consistency."""
    
    def __init__(self, judge_service):
        """Initialize validator.
        
        Args:
            judge_service: LLM judge service instance
        """
        self.judge_service = judge_service
        self.validation_results = []
    
    def validate_judge_consistency(self,
                                   test_cases: List[Dict[str, Any]],
                                   num_runs: int = 5) -> Dict[str, Any]:
        """Validate judge consistency by running same evaluations multiple times.
        
        Industry Standard: OpenAI Evals validates judge reliability
        
        Args:
            test_cases: List of test cases to evaluate
            num_runs: Number of times to run each evaluation
            
        Returns:
            Validation results with variance metrics
        """
        logger.info(f"Validating judge consistency with {len(test_cases)} test cases, {num_runs} runs each")
        
        results = {
            "test_cases_evaluated": len(test_cases),
            "runs_per_case": num_runs,
            "consistency_scores": [],
            "high_variance_cases": [],
            "overall_consistency": 0.0
        }
        
        for test_case in test_cases:
            scores = []
            
            # Run evaluation multiple times
            for run in range(num_runs):
                try:
                    # Call judge service
                    judge_result = self.judge_service.judge_helpfulness(
                        input_text=test_case.get('input', ''),
                        output_text=test_case.get('output', '')
                    )
                    scores.append(judge_result.get('score', 0))
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.error(f"Judge evaluation failed: {e}")
                    scores.append(0)
            
            # Calculate variance
            if scores:
                mean_score = statistics.mean(scores)
                variance = statistics.variance(scores) if len(scores) > 1 else 0
                std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
                
                consistency_score = 1 - min(std_dev / (mean_score + 0.01), 1.0)  # Higher is more consistent
                
                case_result = {
                    "test_case_id": test_case.get('id', 'unknown'),
                    "scores": scores,
                    "mean": mean_score,
                    "variance": variance,
                    "std_dev": std_dev,
                    "consistency_score": consistency_score
                }
                
                results["consistency_scores"].append(case_result)
                
                # Flag high variance cases
                if std_dev > 0.3:  # Threshold for concerning variance
                    results["high_variance_cases"].append(case_result)
        
        # Calculate overall consistency
        if results["consistency_scores"]:
            results["overall_consistency"] = statistics.mean(
                [cs["consistency_score"] for cs in results["consistency_scores"]]
            )
        
        # Generate recommendations
        results["recommendations"] = self._generate_consistency_recommendations(results)
        
        self.validation_results.append(results)
        logger.info(f"Judge consistency validation complete. Overall consistency: {results['overall_consistency']:.2f}")
        
        return results
    
    def compare_judge_to_human(self,
                               human_labels: List[Dict[str, Any]],
                               judge_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare judge scores to human expert labels.
        
        Industry Standard: Validate LLM judge against human agreement
        
        Args:
            human_labels: List of human expert evaluations
            judge_evaluations: List of judge evaluations for same cases
            
        Returns:
            Agreement metrics
        """
        logger.info(f"Comparing judge to human labels ({len(human_labels)} cases)")
        
        if len(human_labels) != len(judge_evaluations):
            raise ValueError("Human labels and judge evaluations must have same length")
        
        agreements = []
        disagreements = []
        
        for human, judge in zip(human_labels, judge_evaluations):
            human_score = human.get('score', 0)
            judge_score = judge.get('score', 0)
            
            # Calculate agreement (allow some tolerance)
            diff = abs(human_score - judge_score)
            agreement = 1 - min(diff, 1.0)
            
            if diff < 0.2:  # Close enough
                agreements.append({
                    "case_id": human.get('id'),
                    "human_score": human_score,
                    "judge_score": judge_score,
                    "agreement": agreement
                })
            else:
                disagreements.append({
                    "case_id": human.get('id'),
                    "human_score": human_score,
                    "judge_score": judge_score,
                    "difference": diff
                })
        
        agreement_rate = len(agreements) / len(human_labels) if human_labels else 0
        
        results = {
            "total_cases": len(human_labels),
            "agreements": len(agreements),
            "disagreements": len(disagreements),
            "agreement_rate": agreement_rate,
            "disagreement_details": disagreements[:5],  # Show first 5
            "overall_correlation": self._calculate_correlation(human_labels, judge_evaluations)
        }
        
        results["recommendations"] = self._generate_agreement_recommendations(results)
        
        logger.info(f"Judge-human agreement: {agreement_rate*100:.1f}%")
        
        return results
    
    def validate_judge_calibration(self,
                                   test_cases: List[Dict[str, Any]],
                                   expected_distribution: Dict[str, float] = None) -> Dict[str, Any]:
        """Validate that judge scores follow expected distribution.
        
        Args:
            test_cases: Test cases to evaluate
            expected_distribution: Expected score distribution (e.g., {"low": 0.2, "medium": 0.6, "high": 0.2})
            
        Returns:
            Calibration results
        """
        logger.info(f"Validating judge calibration on {len(test_cases)} test cases")
        
        scores = []
        for test_case in test_cases:
            try:
                judge_result = self.judge_service.judge_helpfulness(
                    input_text=test_case.get('input', ''),
                    output_text=test_case.get('output', '')
                )
                scores.append(judge_result.get('score', 0))
            except Exception as e:
                logger.error(f"Judge evaluation failed: {e}")
        
        if not scores:
            return {"error": "No scores collected"}
        
        # Analyze distribution
        low_scores = len([s for s in scores if s < 0.4])
        medium_scores = len([s for s in scores if 0.4 <= s < 0.7])
        high_scores = len([s for s in scores if s >= 0.7])
        
        actual_distribution = {
            "low": low_scores / len(scores),
            "medium": medium_scores / len(scores),
            "high": high_scores / len(scores)
        }
        
        results = {
            "total_cases": len(scores),
            "actual_distribution": actual_distribution,
            "expected_distribution": expected_distribution or {"note": "No expected distribution provided"},
            "mean_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
        }
        
        # Check for concerning patterns
        if actual_distribution["low"] > 0.7:
            results["warning"] = "Judge appears overly critical (70%+ low scores)"
        elif actual_distribution["high"] > 0.7:
            results["warning"] = "Judge appears overly lenient (70%+ high scores)"
        
        results["recommendations"] = self._generate_calibration_recommendations(results)
        
        return results
    
    def _calculate_correlation(self, human_labels: List[Dict], judge_evaluations: List[Dict]) -> float:
        """Calculate Pearson correlation between human and judge scores."""
        human_scores = [h.get('score', 0) for h in human_labels]
        judge_scores = [j.get('score', 0) for j in judge_evaluations]
        
        if len(human_scores) < 2:
            return 0.0
        
        # Simple correlation calculation
        mean_human = statistics.mean(human_scores)
        mean_judge = statistics.mean(judge_scores)
        
        numerator = sum((h - mean_human) * (j - mean_judge) 
                       for h, j in zip(human_scores, judge_scores))
        
        denominator = (
            sum((h - mean_human) ** 2 for h in human_scores) ** 0.5 *
            sum((j - mean_judge) ** 2 for j in judge_scores) ** 0.5
        )
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _generate_consistency_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on consistency validation."""
        recommendations = []
        
        if results["overall_consistency"] < 0.7:
            recommendations.append("❌ Judge consistency is low. Consider using temperature=0 for deterministic outputs.")
            recommendations.append("Consider using a more powerful model for judging (e.g., GPT-4 instead of GPT-3.5)")
        
        if results["high_variance_cases"]:
            recommendations.append(f"⚠️ {len(results['high_variance_cases'])} cases show high variance. Review these cases manually.")
            recommendations.append("Consider adding more specific evaluation criteria to reduce ambiguity")
        
        if results["overall_consistency"] >= 0.85:
            recommendations.append("✅ Judge consistency is excellent. Scores are reliable.")
        
        return recommendations
    
    def _generate_agreement_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on human agreement."""
        recommendations = []
        
        if results["agreement_rate"] < 0.6:
            recommendations.append("❌ Low agreement with human labels. Judge may not be well-calibrated.")
            recommendations.append("Review disagreement cases to identify patterns")
            recommendations.append("Consider fine-tuning judge prompts or using human feedback for training")
        
        elif results["agreement_rate"] < 0.8:
            recommendations.append("⚠️ Moderate agreement. Some calibration improvements needed.")
        
        else:
            recommendations.append("✅ High agreement with human labels. Judge is well-calibrated.")
        
        return recommendations
    
    def _generate_calibration_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on calibration validation."""
        recommendations = []
        
        if "warning" in results:
            recommendations.append(f"⚠️ {results['warning']}")
            recommendations.append("Adjust judge prompt to provide more balanced evaluations")
        
        if results["std_dev"] < 0.1:
            recommendations.append("⚠️ Very low variance in scores. Judge may not be discriminating enough.")
        
        return recommendations

    # ------------------------------------------------------------------
    # Adversarial Validation (Phase 3 — Fix #10)
    # ------------------------------------------------------------------

    def adversarial_validate(self, synthesis_report: str, evidence: dict,
                              threshold: float = 0.7,
                              llm_caller=None) -> Dict[str, Any]:
        """Cross-reference synthesis report claims against evidence.

        Step 1: Extract testable factual claims via LLM.
        Step 2: Cross-reference each claim against evidence.
        Step 3: Score and optionally produce corrected report.

        Args:
            synthesis_report: The full synthesis report text.
            evidence: Dict of evidence sources {label: text}.
            threshold: Survival rate below which report is corrected.
            llm_caller: Optional callable(prompt, model, max_tokens) -> str.
                        Defaults to self._default_llm_caller.

        Returns:
            Dict with claim_survival_rate, total_claims, supported_claims,
            flagged_claims, corrected_report, should_replace.
        """
        logger.info("Starting adversarial validation")
        caller = llm_caller or self._default_llm_caller

        # Step 1 — Claim extraction
        claims = self._extract_claims(synthesis_report, caller)
        logger.info(f"Extracted {len(claims)} claims")

        if not claims:
            return {
                "claim_survival_rate": 1.0,
                "total_claims": 0,
                "supported_claims": 0,
                "flagged_claims": [],
                "corrected_report": synthesis_report,
                "should_replace": False,
            }

        # Step 2 — Cross-reference each claim
        verdicts = []
        for claim in claims:
            verdict = self._check_claim(claim, evidence, caller)
            verdicts.append(verdict)

        supported = [v for v in verdicts if v["verdict"] == "supported"]
        survival_rate = len(supported) / len(verdicts) if verdicts else 1.0

        # Step 3 — Generate corrected report if needed
        corrected = synthesis_report
        should_replace = survival_rate < threshold
        if should_replace:
            corrected = self._generate_corrected_report(
                synthesis_report, verdicts, caller)

        result = {
            "claim_survival_rate": round(survival_rate, 3),
            "total_claims": len(claims),
            "supported_claims": len(supported),
            "flagged_claims": [
                v for v in verdicts if v["verdict"] != "supported"],
            "corrected_report": corrected,
            "should_replace": should_replace,
        }

        logger.info(
            f"Adversarial validation complete: "
            f"{result['supported_claims']}/{result['total_claims']} claims "
            f"survived ({result['claim_survival_rate']:.0%})")
        return result

    def _extract_claims(self, report: str, caller) -> List[str]:
        """Extract testable factual claims from a report via LLM."""
        import json as _json
        import re as _re

        prompt = (
            "Extract all testable factual claims from this report. "
            "Return ONLY a JSON array of strings, no explanation. "
            "Only include claims that reference specific data, numbers, "
            "or verifiable facts.\n\n"
            f"Report:\n{report[:3000]}"
        )
        response = caller(prompt, model="qwen3:8b", max_tokens=1000)

        # Strip Qwen3 <think>...</think> reasoning tags
        cleaned = _re.sub(
            r"<think>.*?</think>", "", response, flags=_re.DOTALL
        ).strip()

        # Try JSON parse first
        try:
            parsed = _json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(c) for c in parsed if c]
        except (_json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON array in the response
        json_match = _re.search(r'\[.*\]', cleaned, _re.DOTALL)
        if json_match:
            try:
                parsed = _json.loads(json_match.group())
                if isinstance(parsed, list):
                    return [str(c) for c in parsed if c]
            except (_json.JSONDecodeError, TypeError):
                pass

        # Fallback: treat each non-empty line as a claim
        lines = [
            line.strip().lstrip("- •*0123456789.)")
            for line in cleaned.split("\n")
            if line.strip() and len(line.strip()) > 15
        ]
        return lines[:20]  # Cap at 20 claims

    def _check_claim(self, claim: str, evidence: dict, caller) -> dict:
        """Cross-reference a single claim against evidence."""
        import re as _re

        evidence_text = "\n".join(
            f"[{k}]: {str(v)[:300]}" for k, v in evidence.items()
        )
        prompt = (
            f"Is this claim supported by the evidence?\n\n"
            f"CLAIM: {claim}\n\n"
            f"EVIDENCE:\n{evidence_text[:2000]}\n\n"
            "Reply with exactly one of:\n"
            "SUPPORTED — evidence confirms this claim\n"
            "UNSUPPORTED — no evidence found for this claim\n"
            "CONTRADICTED — evidence directly contradicts this claim\n\n"
            "Then explain in one sentence why."
        )
        response = caller(prompt, model="qwen3:8b", max_tokens=200)
        # Strip Qwen3 <think>...</think> reasoning tags
        cleaned = _re.sub(
            r"<think>.*?</think>", "", response, flags=_re.DOTALL
        ).strip()
        verdict = "unsupported"
        upper = cleaned.upper()[:50]
        if "SUPPORTED" in upper and "UNSUPPORTED" not in upper:
            verdict = "supported"
        elif "CONTRADICTED" in upper:
            verdict = "contradicted"
        return {"claim": claim, "verdict": verdict, "reason": cleaned}

    def _generate_corrected_report(self, original: str, verdicts: list,
                                    caller) -> str:
        """Rewrite report with unsupported/contradicted claims fixed."""
        flagged = [v for v in verdicts if v["verdict"] != "supported"]
        if not flagged:
            return original

        flags_text = "\n".join(
            f"- {v['claim']} [{v['verdict']}]: {v['reason']}"
            for v in flagged
        )
        prompt = (
            "Rewrite this report, removing or marking claims that were "
            "not supported by evidence.\n\n"
            f"ORIGINAL REPORT:\n{original[:2000]}\n\n"
            f"FLAGGED CLAIMS:\n{flags_text}\n\n"
            "Rewrite the report with unsupported claims removed and "
            "contradicted claims corrected."
        )
        return caller(prompt, model="qwen3:8b", max_tokens=1500)

    @staticmethod
    def _default_llm_caller(prompt: str, model: str = "qwen3:8b",
                             max_tokens: int = 500) -> str:
        """Default LLM caller using the ABM module's call_ollama."""
        try:
            from src.abm.llm_utils import call_ollama
            return call_ollama(prompt, model=model, max_tokens=max_tokens)
        except ImportError:
            # Fallback: try orchestrator's call_ollama
            try:
                import sys
                import os
                root = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
                if root not in sys.path:
                    sys.path.insert(0, root)
                from orchestrator import call_ollama as orch_call
                return orch_call(prompt, model=model, max_tokens=max_tokens)
            except ImportError:
                return "[ERROR] No LLM caller available"

