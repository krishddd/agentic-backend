import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Bug 3 fix: guard optional imports that may not exist in all configurations
try:
    from src.retriever import RetrieverManager
    from src.model import Model
    from src.configuration import ConfigurationManager
    from src.agent import create_enhanced_rag_agent_executor
    from src.test_data_generation import TestDataGeneration
    _IMPORTS_AVAILABLE = True
except ImportError as _import_err:
    logger.warning(f"Optional evaluation imports unavailable: {_import_err}")
    _IMPORTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """Comprehensive retrieval quality evaluation system for RAG pipelines."""

    def __init__(self, retriever, config_dict: Dict[str, Any]):
        self.retriever = retriever
        self.config_dict = config_dict
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    def load_evaluation_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """Loads the ground truth dataset from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                dataset = json.load(f)
                logger.info(f"Loaded {len(dataset)} evaluation samples from {filepath}")
                return dataset
        except FileNotFoundError:
            logger.error(f"Evaluation dataset not found at: {filepath}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in: {filepath}")
            return []

    def calculate_basic_metrics(self, retrieved_docs: List[str], expected_phrases: List[str]) -> Dict[str, float]:
        """Calculates basic Context Precision and Context Recall."""
        if not retrieved_docs:
            return {"precision": 0.0, "recall": 0.0}

        retrieved_content = " ".join(retrieved_docs).lower()

        relevant_docs_count = sum(1 for doc in retrieved_docs if any(p.lower() in doc.lower() for p in expected_phrases))
        precision = relevant_docs_count / len(retrieved_docs) if retrieved_docs else 0.0

        found_phrases = sum(1 for phrase in expected_phrases if phrase.lower() in retrieved_content)
        recall = found_phrases / len(expected_phrases) if expected_phrases else 0.0

        return {"precision": precision, "recall": recall}

    def calculate_advanced_metrics(self, retrieved_docs: List[str], expected_docs: List[str]) -> Dict[str, float]:
        """Calculates advanced metrics using semantic similarity."""
        default_metrics = {"semantic_precision": 0.0, "semantic_recall": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
        if not retrieved_docs or not expected_docs:
            return default_metrics

        try:
            all_docs = retrieved_docs + expected_docs
            tfidf_matrix = self.vectorizer.fit_transform(all_docs)
            retrieved_vectors, expected_vectors = tfidf_matrix[:len(retrieved_docs)], tfidf_matrix[len(retrieved_docs):]
            
            similarity_matrix = cosine_similarity(retrieved_vectors, expected_vectors)

            return {
                "semantic_precision": np.mean(np.max(similarity_matrix, axis=1)),
                "semantic_recall": np.mean(np.max(similarity_matrix, axis=0)),
                "mrr": self._calculate_mrr(similarity_matrix),
                "ndcg_at_k": self._calculate_ndcg_at_k(similarity_matrix, k=5)
            }
        except Exception as e:
            logger.warning(f"Error calculating advanced metrics: {e}")
            return default_metrics

    def _calculate_mrr(self, sim_matrix: np.ndarray) -> float:
        """Calculate Mean Reciprocal Rank."""
        ranks = [1.0 / (np.where(np.argsort(-sim_matrix[:, i]) == np.argmax(sim_matrix[:, i]))[0][0] + 1) for i in range(sim_matrix.shape[1])]
        return np.mean(ranks) if ranks else 0.0

    def _calculate_ndcg_at_k(self, sim_matrix: np.ndarray, k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""
        ndcg_scores = []
        for i in range(sim_matrix.shape[1]):
            relevances = sim_matrix[np.argsort(-sim_matrix[:, i]), i]
            dcg = np.sum([(2**rel - 1) / np.log2(j + 2) for j, rel in enumerate(relevances[:k])])
            ideal_relevances = np.sort(sim_matrix[:, i])[::-1]
            idcg = np.sum([(2**rel - 1) / np.log2(j + 2) for j, rel in enumerate(ideal_relevances[:k])])
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def calculate_retrieval_metrics(self, retrieved_docs: List[str], expected_phrases: List[str], expected_docs: List[str] = None) -> Dict[str, float]:
        """Calculate comprehensive retrieval metrics for a single question."""
        metrics = {
            **self.calculate_basic_metrics(retrieved_docs, expected_phrases),
        }
        if expected_docs:
            metrics.update(self.calculate_advanced_metrics(retrieved_docs, expected_docs))
        
        precision, recall = metrics.get("precision", 0.0), metrics.get("recall", 0.0)
        metrics["f1_score"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return metrics

    def evaluate_retriever_comprehensive(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        """Run comprehensive evaluation on the entire dataset."""
        results = []
        logger.info(f"Starting retrieval evaluation on {len(dataset)} samples...")
        for i, item in enumerate(dataset):
            try:
                retrieved_docs = self.retriever.invoke(item["question"])
                content = [doc.page_content for doc in retrieved_docs]
                metrics = self.calculate_retrieval_metrics(
                    content, item.get("expected_retrieved_phrases", []), item.get("expected_documents", [])
                )
                results.append({
                    "Question": item["question"], **metrics
                })
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
        return pd.DataFrame(results)

    def generate_evaluation_report(self, results_df: pd.DataFrame, output_dir: str = "evaluation_results"):
        """Generates a comprehensive report with metrics and visualizations."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save raw results
        results_df.to_csv(os.path.join(output_dir, "retrieval_evaluation_results.csv"), index=False)

        # 2. Summary statistics
        summary = results_df.describe()
        summary.to_csv(os.path.join(output_dir, "retrieval_summary_statistics.csv"))

        # 3. Visualizations
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=results_df.drop(columns=['Question'], errors='ignore'))
        plt.title("Distribution of Retrieval Metrics")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "retrieval_metrics_distribution.png"))
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(results_df.drop(columns=['Question'], errors='ignore').corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of Retrieval Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "retrieval_metrics_correlation.png"))
        plt.close()

        logger.info(f"Evaluation report saved to {os.path.abspath(output_dir)}")


class GenerationEvaluator:
    """Evaluates the quality of the generated response from a RAG pipeline."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = JsonOutputParser()

    def _evaluate_with_llm(self, prompt_template: PromptTemplate, input_data: Dict[str, Any]) -> Dict:
        """Helper to run evaluation chains."""
        try:
            prompt = prompt_template.format(**input_data)
            chain = self.llm | self.parser
            return chain.invoke(prompt)
        except Exception as e:
            logger.error(f"LLM-based evaluation failed: {e}")
            return {"score": 0.0, "reasoning": "Evaluation failed."}

    def evaluate_faithfulness(self, question: str, retrieved_contexts: List[str], generated_answer: str) -> Dict:
        """Evaluates if the answer is factually consistent with the retrieved contexts."""
        template = """
        Evaluate the faithfulness of the generated answer based on the provided context.
        Score 1 if the answer is fully supported by the context, and 0 otherwise.
        Provide a brief reasoning for your score.

        Context: {context}
        Question: {question}
        Answer: {answer}

        Format your response as a JSON object with "score" and "reasoning" keys.
        """
        prompt = PromptTemplate.from_template(template)
        context_str = "\n".join(retrieved_contexts)
        return self._evaluate_with_llm(prompt, {"context": context_str, "question": question, "answer": generated_answer})

    def evaluate_answer_relevancy(self, question: str, generated_answer: str) -> Dict:
        """Evaluates if the answer is relevant to the question."""
        template = """
        Evaluate the relevancy of the generated answer to the question.
        Score 1 if the answer is relevant, and 0 otherwise.
        Provide a brief reasoning for your score.

        Question: {question}
        Answer: {answer}

        Format your response as a JSON object with "score" and "reasoning" keys.
        """
        prompt = PromptTemplate.from_template(template)
        return self._evaluate_with_llm(prompt, {"question": question, "answer": generated_answer})

    def evaluate_answer_correctness(self, ground_truth_answer: str, generated_answer: str) -> Dict:
        """Evaluates if the generated answer is correct compared to a ground truth."""
        template = """
        Evaluate the correctness of the generated answer compared to the ground truth answer.
        Score 1 if the answer is correct, and 0 otherwise.
        Provide a brief reasoning for your score.

        Ground Truth Answer: {ground_truth}
        Generated Answer: {generated_answer}

        Format your response as a JSON object with "score" and "reasoning" keys.
        """
        prompt = PromptTemplate.from_template(template)
        return self._evaluate_with_llm(prompt, {"ground_truth": ground_truth_answer, "generated_answer": generated_answer})


def run_full_pipeline_evaluation(agent_executor, dataset_path: str, llm: BaseChatModel) -> pd.DataFrame:
    """
    Runs a full evaluation of the RAG pipeline, including retrieval and generation.
    """
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    generation_evaluator = GenerationEvaluator(llm)
    results = []

    for item in dataset:
        question = item["question"]
        ground_truth = item["ground_truth_answer"]

        agent_response = agent_executor.invoke({"input": question})
        generated_answer = agent_response.get("output", "")
        retrieved_docs = agent_response.get("intermediate_steps", [])
        
        retrieved_contexts = [str(doc) for doc in retrieved_docs]

        faithfulness = generation_evaluator.evaluate_faithfulness(question, retrieved_contexts, generated_answer)
        relevancy = generation_evaluator.evaluate_answer_relevancy(question, generated_answer)
        correctness = generation_evaluator.evaluate_answer_correctness(ground_truth, generated_answer)

        results.append({
            "question": question,
            "generated_answer": generated_answer,
            "faithfulness_score": faithfulness['score'],
            "faithfulness_reasoning": faithfulness['reasoning'],
            "relevancy_score": relevancy['score'],
            "relevancy_reasoning": relevancy['reasoning'],
            "correctness_score": correctness['score'],
            "correctness_reasoning": correctness['reasoning'],
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # --- Example Usage ---
    
    # 1. Initialize pipeline components
    config_manager = ConfigurationManager()
    config_dict = config_manager.configurations()
    model_obj = Model(config_dict)
    llm = model_obj.load_ollama_model()

    # 2. Generate a new test dataset from your documents
    print("--- Generating Test Data ---")
    test_data_generator = TestDataGeneration(model=llm)
    dataset_path = test_data_generator.generator(test_size=10) # Generate 10 QA pairs

    # 3. Create the agent executor
    agent = create_enhanced_rag_agent_executor(
        memory_type="summary_buffer",
        enable_performance_tracking=True
    )

    # 4. Run the full pipeline evaluation
    evaluation_results_df = run_full_pipeline_evaluation(agent, dataset_path, llm)

    # 5. Display and save results
    print("\n--- Generation Evaluation Results ---")
    print(evaluation_results_df)
    evaluation_results_df.to_csv("generation_evaluation_results.csv", index=False)
    print("\nEvaluation results saved to generation_evaluation_results.csv")
