"""
Intermediate Step Validation

Validates each tool call and action in the agent workflow.
Industry standard: LangSmith tool chain evaluation
"""
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


class ValidationStatus(Enum):
    """Validation status for each step."""
    CORRECT = "correct"  # Right tool, right params, succeeded
    SUBOPTIMAL = "suboptimal"  # Worked but not optimal
    WRONG_TOOL = "wrong_tool"  # Wrong tool selected
    WRONG_PARAMS = "wrong_params"  # Right tool, wrong parameters
    FAILED = "failed"  # Execution failed
    UNNECESSARY = "unnecessary"  # Redundant action


@dataclass
class StepValidation:
    """Validation result for single step."""
    step_num: int
    tool_name: str
    parameters: Dict[str, Any]
    result_success: bool
    status: ValidationStatus
    score: float  # 0-1
    issues: List[str]
    recommendations: List[str]
    explanation: str


class IntermediateStepValidator:
    """
    Validates each intermediate step in agent execution.
    
    Industry standard: LangSmith approach
    Checks if each tool selection and parameter passing was correct.
    """
    
    def __init__(self):
        # Define validation rules for each tool
        self.tool_rules = {
            "sheets_read": {
                "required_params": ["sheet_id", "range"],
                "valid_before": [],  # Can be called anytime
                "optimal_after": []  # Should be first step usually
            },
            "gmail_send": {
                "required_params": ["to", "subject", "body"],
                "valid_before": [],
                "optimal_after": ["llm_generate"]  # Should generate email first
            },
            "llm_generate": {
                "required_params": ["prompt"],
                "valid_before": ["gmail_send", "sheets_update"],
                "optimal_after": ["sheets_read", "rag_retrieve"]  # Should have context first
            },
            "rag_retrieve": {
                "required_params": ["query"],
                "valid_before": ["llm_generate"],
                "optimal_after": ["sheets_read"]
            },
            "sheets_update": {
                "required_params": ["sheet_id", "range", "values"],
                "valid_before": [],
                "optimal_after": ["gmail_send", "llm_generate"]  # Should update after actions
            },
            "calendar_generate": {
                "required_params": ["event_details"],
                "valid_before": [],
                "optimal_after": ["sheets_read"]
            }
        }
    
    def validate_step(
        self,
        step_num: int,
        tool_name: str,
        parameters: Dict[str, Any],
        result_success: bool,
        previous_steps: List[Dict[str, str]] = None
    ) -> StepValidation:
        """
        Validate single step.
        
        Args:
            step_num: Step number in sequence
            tool_name: Tool that was called
            parameters: Parameters passed to tool
            result_success: Did the tool execution succeed?
            previous_steps: List of previous tool calls
        
        Returns:
            Step validation result
        """
        issues = []
        recommendations = []
        score = 1.0
        status = ValidationStatus.CORRECT
        
        # 1. Check if tool exists in our rules
        if tool_name not in self.tool_rules:
            issues.append(f"Unknown tool: {tool_name}")
            score *= 0.7
            status = ValidationStatus.SUBOPTIMAL
        else:
            rules = self.tool_rules[tool_name]
            
            # 2. Validate required parameters
            missing_params = [
                p for p in rules["required_params"]
                if p not in parameters or not parameters[p]
            ]
            if missing_params:
                issues.append(f"Missing required parameters: {missing_params}")
                recommendations.append(f"Add parameters: {', '.join(missing_params)}")
                score *= 0.6
                status = ValidationStatus.WRONG_PARAMS
            
            # 3. Check ordering - should this tool come after certain others?
            if previous_steps and rules["optimal_after"]:
                prev_tools = [s["tool"] for s in previous_steps]
                expected_prev = rules["optimal_after"]
                
                # Check if any expected predecessors were called
                if not any(tool in prev_tools for tool in expected_prev):
                    issues.append(
                        f"{tool_name} usually should be called after: {expected_prev}"
                    )
                    recommendations.append(
                        f"Consider calling {expected_prev} before {tool_name}"
                    )
                    score *= 0.9
                    if status == ValidationStatus.CORRECT:
                        status = ValidationStatus.SUBOPTIMAL
            
            # 4. Check if calling this tool is valid at this point
            if previous_steps and rules["valid_before"]:
                # Check if tools that should come AFTER this one haven't been called yet
                prev_tools = [s["tool"] for s in previous_steps]
                already_called = [t for t in rules["valid_before"] if t in prev_tools]
                
                if already_called:
                    issues.append(
                        f"{tool_name} should be called BEFORE {already_called}, "
                        f"but they were already called"
                    )
                    score *= 0.8
                    status = ValidationStatus.WRONG_TOOL
        
        # 5. Check execution result
        if not result_success:
            issues.append("Step execution failed")
            recommendations.append("Check error logs and retry with correct parameters")
            score *= 0.5
            status = ValidationStatus.FAILED
        
        # 6. Check for redundancy
        if previous_steps:
            same_tool_count = sum(
                1 for s in previous_steps
                if s.get("tool") == tool_name
            )
            if same_tool_count >= 2:
                issues.append(
                    f"Redundant: {tool_name} called {same_tool_count + 1} times"
                )
                recommendations.append(
                    "Optimize to reduce redundant tool calls"
                )
                score *= 0.85
                if status == ValidationStatus.CORRECT:
                    status = ValidationStatus.UNNECESSARY
        
        # Generate explanation
        if status == ValidationStatus.CORRECT:
            explanation = f"✓ {tool_name} called correctly with proper parameters and ordering"
        elif status == ValidationStatus.SUBOPTIMAL:
            explanation = f"○ {tool_name} worked but could be optimized"
        elif status == ValidationStatus.WRONG_TOOL:
            explanation = f"⚠ {tool_name} may not be the right tool at this point"
        elif status == ValidationStatus.WRONG_PARAMS:
            explanation = f"✗ {tool_name} called with incorrect parameters"
        elif status == ValidationStatus.FAILED:
            explanation = f"✗ {tool_name} execution failed"
        else:
            explanation = f"⚠ {tool_name} may be unnecessary"
        
        return StepValidation(
            step_num=step_num,
            tool_name=tool_name,
            parameters=parameters,
            result_success=result_success,
            status=status,
            score=score,
            issues=issues,
            recommendations=recommendations,
            explanation=explanation
        )
    
    def validate_workflow(
        self,
        steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate complete workflow.
        
        Args:
            steps: List of steps with 'tool', 'parameters', 'success'
        
        Returns:
            Complete workflow validation
        """
        validations = []
        previous_steps = []
        
        for i, step in enumerate(steps):
            tool_name = step.get("tool", "unknown")
            parameters = step.get("parameters", {})
            success = step.get("success", True)
            
            validation = self.validate_step(
                step_num=i + 1,
                tool_name=tool_name,
                parameters=parameters,
                result_success=success,
                previous_steps=previous_steps
            )
            
            validations.append(validation)
            previous_steps.append({"tool": tool_name})
        
        # Calculate overall scores
        total_score = (
            sum(v.score for v in validations) / len(validations)
            if validations else 0.0
        )
        
        correct_count = sum(
            1 for v in validations
            if v.status == ValidationStatus.CORRECT
        )
        
        failed_count = sum(
            1 for v in validations
            if v.status == ValidationStatus.FAILED
        )
        
        suboptimal_count = sum(
            1 for v in validations
            if v.status in [ValidationStatus.SUBOPTIMAL, ValidationStatus.UNNECESSARY]
        )
        
        summary = {
            "total_steps": len(validations),
            "correct_steps": correct_count,
            "suboptimal_steps": suboptimal_count,
            "failed_steps": failed_count,
            "overall_score": total_score,
            "validations": [
                {
                    "step_num": v.step_num,
                    "tool": v.tool_name,
                    "status": v.status.value,
                    "score": v.score,
                    "issues": v.issues,
                    "recommendations": v.recommendations,
                    "explanation": v.explanation
                }
                for v in validations
            ],
            "summary": self._generate_summary(validations, total_score)
        }
        
        return summary
    
    def _generate_summary(
        self,
        validations: List[StepValidation],
        total_score: float
    ) -> str:
        """Generate human-readable summary."""
        if total_score >= 0.9:
            status = "✓ Excellent"
        elif total_score >= 0.7:
            status = "○ Good"
        else:
            status = "✗ Needs improvement"
        
        failed = [v for v in validations if v.status == ValidationStatus.FAILED]
        suboptimal = [
            v for v in validations
            if v.status in [ValidationStatus.SUBOPTIMAL, ValidationStatus.UNNECESSARY]
        ]
        
        summary = f"{status} - Overall tool usage score: {total_score:.1%}\n"
        
        if failed:
            summary += f"\n{len(failed)} steps failed execution\n"
        
        if suboptimal:
            summary += f"{len(suboptimal)} steps could be optimized\n"
        
        # Collect unique recommendations
        all_recommendations = set()
        for v in validations:
            all_recommendations.update(v.recommendations)
        
        if all_recommendations:
            summary += "\nRecommendations:\n"
            for rec in list(all_recommendations)[:5]:  # Top 5
                summary += f"  • {rec}\n"
        
        return summary.strip()
    
    def add_tool_rule(
        self,
        tool_name: str,
        required_params: List[str],
        valid_before: List[str] = None,
        optimal_after: List[str] = None
    ):
        """Add validation rule for a new tool."""
        self.tool_rules[tool_name] = {
            "required_params": required_params,
            "valid_before": valid_before or [],
            "optimal_after": optimal_after or []
        }
        logger.info(f"Added validation rule for tool: {tool_name}")
