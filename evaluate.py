"""
Evaluation Experiment: Comparing Chunking Strategies in Document Q&A RAG
Evaluates Strategy A (Fixed-size) vs. Strategy B (Paragraph-aware) on:
- Recall@5
- Mean Reciprocal Rank (MRR)
- Average Retrieval Latency (ms)
- Number of Chunks Generated
- Abstention Accuracy on Out-of-Domain Questions

Outputs results to results/chunking_comparison.csv and displays comparison summary.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Tuple
import pandas as pd

from rag_pipeline import (
    DocumentIngestor,
    DocumentIngestionError,
    Chunker,
    VectorStoreManager,
    Retriever,
    OllamaClient,
    DEFAULT_SIMILARITY_THRESHOLD,
    TOP_K_DEFAULT
)

# Ground-truth evaluation dataset tailored to the NIST AI corpus
# Includes:
# - Normal questions with answers in NIST AI 100-1 (RMF 1.0)
# - Questions with answers in NIST AI 600-1 (Generative AI Profile)
# - Cross-document synthesis questions
# - Out-of-domain / unanswerable questions for abstention testing
EVALUATION_DATASET = [
    {
        "id": "q1",
        "question": "What are the four core functions of the NIST AI Risk Management Framework?",
        "expected_doc": "NIST.AI.100-1.pdf",
        "expected_concepts": ["govern", "map", "measure", "manage"],
        "is_answerable": True,
        "category": "In-Domain Single Doc"
    },
    {
        "id": "q2",
        "question": "What characteristics define trustworthy artificial intelligence according to NIST AI RMF?",
        "expected_doc": "NIST.AI.100-1.pdf",
        "expected_concepts": ["valid", "reliable", "safe", "secure", "transparent", "explainable", "privacy"],
        "is_answerable": True,
        "category": "In-Domain Single Doc"
    },
    {
        "id": "q3",
        "question": "What is the primary role of the GOVERN function in the AI RMF?",
        "expected_doc": "NIST.AI.100-1.pdf",
        "expected_concepts": ["govern", "culture", "risk management", "policies", "accountability"],
        "is_answerable": True,
        "category": "In-Domain Single Doc"
    },
    {
        "id": "q4",
        "question": "What specific risks are introduced by generative artificial intelligence in NIST AI 600-1?",
        "expected_doc": "NIST.AI.600-1.pdf",
        "expected_concepts": ["hallucination", "confabulation", "intellectual property", "cbrn", "synthetic content"],
        "is_answerable": True,
        "category": "In-Domain Single Doc"
    },
    {
        "id": "q5",
        "question": "How does the Generative AI Profile address data privacy and copyright risks?",
        "expected_doc": "NIST.AI.600-1.pdf",
        "expected_concepts": ["copyright", "privacy", "training data", "intellectual property", "pii"],
        "is_answerable": True,
        "category": "In-Domain Single Doc"
    },
    {
        "id": "q6",
        "question": "How does the AI RMF Map function apply to identifying context and risks in generative AI deployment?",
        "expected_doc": None,  # Can match either or both NIST documents
        "expected_concepts": ["map", "context", "risk", "impact", "lifecycle"],
        "is_answerable": True,
        "category": "Cross-Document Synthesis"
    },
    {
        "id": "q7",
        "question": "What is the role of continuous measurement and TEVV in the MEASURE function?",
        "expected_doc": "NIST.AI.100-1.pdf",
        "expected_concepts": ["measure", "tevv", "test", "evaluate", "verify", "validate"],
        "is_answerable": True,
        "category": "In-Domain Single Doc"
    },
    {
        "id": "q8",
        "question": "What risk mitigations does NIST suggest for chemical, biological, radiological, or nuclear (CBRN) risks in generative AI?",
        "expected_doc": "NIST.AI.600-1.pdf",
        "expected_concepts": ["cbrn", "biological", "chemical", "safeguard", "red-teaming"],
        "is_answerable": True,
        "category": "In-Domain Single Doc"
    },
    {
        "id": "q9",
        "question": "What is the traditional recipe and baking temperature for classic sourdough baguettes?",
        "expected_doc": None,
        "expected_concepts": [],
        "is_answerable": False,
        "category": "Out-of-Domain Abstention"
    },
    {
        "id": "q10",
        "question": "Who won the FIFA World Cup final in 2022 and what was the score during the penalty shootout?",
        "expected_doc": None,
        "expected_concepts": [],
        "is_answerable": False,
        "category": "Out-of-Domain Abstention"
    }
]


def is_passage_relevant(passage: Dict[str, Any], query_spec: Dict[str, Any]) -> bool:
    """
    Determines whether a candidate passage is genuinely relevant to the evaluation query.
    Requires matching document provenance (if specified) and key semantic concept anchors.
    """
    if not query_spec["is_answerable"]:
        return False

    # Check document match if specified
    if query_spec.get("expected_doc"):
        if passage["filename"] != query_spec["expected_doc"]:
            return False

    text_lower = passage["text"].lower()
    expected_concepts = query_spec["expected_concepts"]
    
    # Check concept presence (requires at least 2 key concepts or 50% of expected concepts)
    matched_concepts = [c for c in expected_concepts if c in text_lower]
    threshold_count = min(2, len(expected_concepts))
    return len(matched_concepts) >= threshold_count


def evaluate_strategy(
    strategy_name: str,
    chunks: List[Dict[str, Any]],
    vector_mgr: VectorStoreManager,
    strategy_code: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    top_k: int = TOP_K_DEFAULT
) -> Dict[str, Any]:
    """
    Evaluates a specific chunking strategy across the test dataset.
    Measures:
    - Recall@5
    - MRR (Mean Reciprocal Rank)
    - Average Latency (ms)
    - Abstention Accuracy
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {strategy_name} ({len(chunks)} chunks)...")
    print(f"{'='*60}")

    # Build or load FAISS index
    start_embed = time.perf_counter()
    index, chunks = vector_mgr.build_and_save_index(chunks, strategy=strategy_code)
    embed_time = time.perf_counter() - start_embed
    print(f"Index built and persisted in {embed_time:.2f}s.")

    retriever = Retriever(
        vector_store_mgr=vector_mgr,
        index=index,
        chunks=chunks,
        similarity_threshold=similarity_threshold
    )

    recalls = []
    reciprocal_ranks = []
    latencies = []
    abstention_checks = []
    query_details = []

    answerable_queries = [q for q in EVALUATION_DATASET if q["is_answerable"]]
    unanswerable_queries = [q for q in EVALUATION_DATASET if not q["is_answerable"]]

    for q in EVALUATION_DATASET:
        t0 = time.perf_counter()
        passages = retriever.retrieve(q["question"], top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        if q["is_answerable"]:
            found_relevant = False
            first_rel_rank = 0

            for rank, p in enumerate(passages, start=1):
                if is_passage_relevant(p, q):
                    found_relevant = True
                    first_rel_rank = rank
                    break

            recalls.append(1.0 if found_relevant else 0.0)
            reciprocal_ranks.append(1.0 / first_rel_rank if found_relevant else 0.0)

            query_details.append({
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "latency_ms": round(latency_ms, 2),
                "top_score": passages[0]["score"] if passages else 0.0,
                "recall@5": 1.0 if found_relevant else 0.0,
                "first_rel_rank": first_rel_rank,
                "abstain_correct": None
            })
        else:
            # Out-of-domain abstention evaluation:
            # Passes if highest similarity score is below threshold or no passage qualifies
            qualifying = [p for p in passages if p["score"] >= similarity_threshold]
            top_score = passages[0]["score"] if passages else 0.0
            correctly_abstained = (len(qualifying) == 0)

            abstention_checks.append(1.0 if correctly_abstained else 0.0)

            query_details.append({
                "id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "latency_ms": round(latency_ms, 2),
                "top_score": top_score,
                "recall@5": None,
                "first_rel_rank": None,
                "abstain_correct": correctly_abstained
            })

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    mean_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    abstention_acc = sum(abstention_checks) / len(abstention_checks) if abstention_checks else 0.0

    print(f"Results for {strategy_name}:")
    print(f"  Total Chunks:           {len(chunks)}")
    print(f"  Recall@5:               {mean_recall:.4f} ({sum(recalls):.0f}/{len(recalls)})")
    print(f"  Mean Reciprocal Rank:   {mean_mrr:.4f}")
    print(f"  Average Latency:        {avg_latency:.2f} ms")
    print(f"  Abstention Accuracy:    {abstention_acc:.2%}")

    return {
        "Strategy": strategy_name,
        "Total Chunks": len(chunks),
        "Recall@5": round(mean_recall, 4),
        "MRR": round(mean_mrr, 4),
        "Avg Latency (ms)": round(avg_latency, 2),
        "Abstention Accuracy": round(abstention_acc, 4),
        "query_details": query_details
    }


def run_experiment(documents_dir: str = os.path.join("data", "documents")):
    """
    Runs the full comparative experiment between Strategy A and Strategy B.
    Saves results to results/chunking_comparison.csv.
    """
    print("=" * 60)
    print("Starting Document Q&A RAG Comparative Evaluation")
    print("=" * 60)

    # 1. Ingestion
    ingestor = DocumentIngestor(documents_dir=documents_dir)
    try:
        pages, stats = ingestor.ingest_documents()
    except DocumentIngestionError as e:
        print(f"\n[ERROR] Ingestion failed: {str(e)}")
        sys.exit(1)

    print(f"Corpus Ingested Successfully:")
    print(f"  Total Files:      {stats['total_files']}")
    print(f"  Total Pages:      {stats['total_pages']}")
    print(f"  Extracted Pages:  {stats['extracted_pages']}")

    # 2. Chunking
    chunks_a = Chunker.chunk_strategy_a_fixed(pages, target_words=400, overlap_words=80)
    chunks_b = Chunker.chunk_strategy_b_paragraph(pages, target_words=400, overlap_words=80)

    # 3. Vector Store Manager
    vector_mgr = VectorStoreManager(vector_store_dir="vector_store")

    # 4. Evaluate Strategy A
    res_a = evaluate_strategy(
        strategy_name="Strategy A (Fixed-Size)",
        chunks=chunks_a,
        vector_mgr=vector_mgr,
        strategy_code="fixed"
    )

    # 5. Evaluate Strategy B
    res_b = evaluate_strategy(
        strategy_name="Strategy B (Paragraph-Aware)",
        chunks=chunks_b,
        vector_mgr=vector_mgr,
        strategy_code="paragraph"
    )

    # 6. Build comparison DataFrame
    comparison_records = [
        {
            "Strategy": res_a["Strategy"],
            "Total_Chunks": res_a["Total Chunks"],
            "Recall@5": res_a["Recall@5"],
            "MRR": res_a["MRR"],
            "Avg_Latency_ms": res_a["Avg Latency (ms)"],
            "Abstention_Accuracy": res_a["Abstention Accuracy"]
        },
        {
            "Strategy": res_b["Strategy"],
            "Total_Chunks": res_b["Total Chunks"],
            "Recall@5": res_b["Recall@5"],
            "MRR": res_b["MRR"],
            "Avg_Latency_ms": res_b["Avg Latency (ms)"],
            "Abstention_Accuracy": res_b["Abstention Accuracy"]
        }
    ]

    df_comp = pd.DataFrame(comparison_records)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "chunking_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Actual experimental results saved to: {csv_path}")

    # 7. Print Comparison Table
    print("\n" + "=" * 60)
    print("COMPARATIVE EVALUATION SUMMARY TABLE")
    print("=" * 60)
    print(df_comp.to_string(index=False))
    print("=" * 60)

    # 8. Automated Strategy Selection Analysis
    score_a = res_a["MRR"] * 0.5 + res_a["Recall@5"] * 0.4 + (res_a["Abstention Accuracy"] * 0.1)
    score_b = res_b["MRR"] * 0.5 + res_b["Recall@5"] * 0.4 + (res_b["Abstention Accuracy"] * 0.1)

    print("\nStrategy Recommendation:")
    if score_b >= score_a:
        print(
            "--> Strategy B (Paragraph-Aware Chunking) is RECOMMENDED.\n"
            "    It preserves natural semantic boundaries while achieving strong retrieval performance\n"
            "    and coherent context for answer generation with fewer fragmented sentences."
        )
    else:
        print(
            "--> Strategy A (Fixed-Size Chunking) is RECOMMENDED.\n"
            "    It achieved higher Recall@5 and MRR across the evaluated questions."
        )

    # Check Ollama status
    print("\nChecking Ollama Status for Live Answer Generation:")
    ollama = OllamaClient()
    ready, msg = ollama.check_connection()
    if ready:
        print(f"  [OK] {msg}")
    else:
        print(f"  [NOTE] {msg}")
        print("  To enable live answering with inline citations, start Ollama (`ollama serve`) and pull `llama3.2`.")


if __name__ == "__main__":
    run_experiment()
