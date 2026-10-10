"""
Document Q&A with Citations - Core RAG Pipeline Module
Includes document ingestion, chunking strategies, vector embeddings,
FAISS indexing, retrieval, and Ollama question answering with source citations.
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pymupdf as fitz
from sentence_transformers import SentenceTransformer
import faiss
import requests

# Default configuration constants
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_SIMILARITY_THRESHOLD = 0.35
TOP_K_DEFAULT = 5
MIN_REQUIRED_PAGES = 50


class DocumentIngestionError(Exception):
    """Custom exception raised during document ingestion failures."""
    pass


class DocumentIngestor:
    """
    Handles PDF discovery, page extraction, page counting,
    OCR warning detection, and metadata preservation.
    """

    def __init__(self, documents_dir: str):
        self.documents_dir = documents_dir

    def get_pdf_files(self) -> List[str]:
        """Finds all PDF files in the specified directory."""
        if not os.path.exists(self.documents_dir):
            raise DocumentIngestionError(
                f"Documents folder does not exist: '{self.documents_dir}'. "
                f"Please create it and place at least {MIN_REQUIRED_PAGES} PDF pages inside."
            )

        pdf_files = [
            f for f in os.listdir(self.documents_dir)
            if f.lower().endswith(".pdf")
        ]
        pdf_paths = [os.path.join(self.documents_dir, f) for f in pdf_files]
        return sorted(pdf_paths)

    def ingest_documents(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extracts text from all PDFs, preserving 1-based page numbers and filenames.
        Validates the 50-page requirement and detects unextractable text.

        Returns:
            pages: List of dictionaries with page metadata and text.
            stats: Summary metrics dictionary.
        """
        pdf_paths = self.get_pdf_files()

        if not pdf_paths:
            raise DocumentIngestionError(
                f"No PDF documents found in '{self.documents_dir}'. "
                f"Please place your PDF files in '{self.documents_dir}' (minimum {MIN_REQUIRED_PAGES} pages total)."
            )

        pages: List[Dict[str, Any]] = []
        total_pages_count = 0
        file_stats: List[Dict[str, Any]] = []
        ocr_warnings: List[str] = []

        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                raise DocumentIngestionError(f"Failed to open PDF '{filename}': {str(e)}")

            num_pages = len(doc)
            total_pages_count += num_pages
            file_extracted_chars = 0
            file_extracted_pages = 0

            for page_idx in range(num_pages):
                page_num = page_idx + 1  # 1-based page number
                page_obj = doc.load_page(page_idx)
                text = page_obj.get_text("text").strip()

                char_count = len(text)
                word_count = len(text.split()) if text else 0
                file_extracted_chars += char_count

                if text:
                    file_extracted_pages += 1

                pages.append({
                    "filename": filename,
                    "filepath": pdf_path,
                    "page_number": page_num,
                    "text": text,
                    "char_count": char_count,
                    "word_count": word_count
                })

            doc.close()

            if file_extracted_chars == 0:
                ocr_warnings.append(
                    f"'{filename}' has 0 extractable text characters across {num_pages} pages. "
                    f"This document appears to be scanned or image-only and may require OCR."
                )

            file_stats.append({
                "filename": filename,
                "total_pages": num_pages,
                "extracted_pages": file_extracted_pages,
                "total_chars": file_extracted_chars
            })

        # Strict requirement check: At least 50 pages across all PDFs
        if total_pages_count < MIN_REQUIRED_PAGES:
            raise DocumentIngestionError(
                f"Corpus page requirement not met: Found {total_pages_count} total PDF page(s) "
                f"across {len(pdf_paths)} file(s), but a minimum of {MIN_REQUIRED_PAGES} pages is required. "
                f"Please add more PDF documents to '{self.documents_dir}'."
            )

        stats = {
            "total_files": len(pdf_paths),
            "total_pages": total_pages_count,
            "extracted_pages": sum(f["extracted_pages"] for f in file_stats),
            "file_stats": file_stats,
            "ocr_warnings": ocr_warnings
        }

        return pages, stats


class Chunker:
    """
    Implements two chunking strategies while preserving source page metadata:
    Strategy A: Fixed-size chunking (~400 words, ~80 words overlap).
    Strategy B: Paragraph-aware chunking (splits on paragraphs, bundles to ~400 words).
    """

    @staticmethod
    def chunk_strategy_a_fixed(
        pages: List[Dict[str, Any]],
        target_words: int = 400,
        overlap_words: int = 80
    ) -> List[Dict[str, Any]]:
        """
        Strategy A: Fixed-size word chunking.
        - Approximately 400 words per chunk.
        - Approximately 80 words overlap.
        - Page boundaries strictly preserved so citations identify the correct source page.
        """
        chunks: List[Dict[str, Any]] = []
        chunk_counter = 0
        step_words = max(1, target_words - overlap_words)

        for page in pages:
            text = page["text"].strip()
            if not text:
                continue

            words = text.split()
            if not words:
                continue

            # If page has fewer or equal to target words, single chunk
            if len(words) <= target_words:
                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"fixed_{chunk_counter}",
                    "strategy": "fixed",
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "text": " ".join(words),
                    "word_count": len(words)
                })
                continue

            # Sliding window over words
            start_idx = 0
            while start_idx < len(words):
                end_idx = min(start_idx + target_words, len(words))
                chunk_words = words[start_idx:end_idx]

                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"fixed_{chunk_counter}",
                    "strategy": "fixed",
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "text": " ".join(chunk_words),
                    "word_count": len(chunk_words)
                })

                if end_idx >= len(words):
                    break
                start_idx += step_words

        return chunks

    @staticmethod
    def chunk_strategy_b_paragraph(
        pages: List[Dict[str, Any]],
        target_words: int = 400,
        overlap_words: int = 80
    ) -> List[Dict[str, Any]]:
        """
        Strategy B: Paragraph-aware chunking.
        - Split text using paragraph boundaries (\n\n+ or double line breaks).
        - Combine paragraphs until target size of ~400 words is reached.
        - Preserves paragraph overlap where practical.
        - Preserves page boundaries and attaches source filename and page number.
        """
        chunks: List[Dict[str, Any]] = []
        chunk_counter = 0

        for page in pages:
            raw_text = page["text"].strip()
            if not raw_text:
                continue

            # Split by double newline or regex paragraph separators
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", raw_text) if p.strip()]

            if not paragraphs:
                continue

            current_paras: List[str] = []
            current_word_count = 0

            for para in paragraphs:
                para_words = len(para.split())

                # If single paragraph exceeds target, break it down using fixed chunks
                if para_words > target_words * 1.5:
                    if current_paras:
                        chunk_counter += 1
                        chunk_text = "\n\n".join(current_paras)
                        chunks.append({
                            "chunk_id": f"para_{chunk_counter}",
                            "strategy": "paragraph",
                            "filename": page["filename"],
                            "page_number": page["page_number"],
                            "text": chunk_text,
                            "word_count": current_word_count
                        })
                        current_paras = []
                        current_word_count = 0

                    # Sub-chunk the oversized paragraph
                    sub_words = para.split()
                    step = max(1, target_words - overlap_words)
                    for s_idx in range(0, len(sub_words), step):
                        e_idx = min(s_idx + target_words, len(sub_words))
                        s_chunk = sub_words[s_idx:e_idx]
                        chunk_counter += 1
                        chunks.append({
                            "chunk_id": f"para_{chunk_counter}",
                            "strategy": "paragraph",
                            "filename": page["filename"],
                            "page_number": page["page_number"],
                            "text": " ".join(s_chunk),
                            "word_count": len(s_chunk)
                        })
                        if e_idx >= len(sub_words):
                            break
                    continue

                if current_word_count + para_words > target_words and current_paras:
                    # Emit current chunk
                    chunk_counter += 1
                    chunk_text = "\n\n".join(current_paras)
                    chunks.append({
                        "chunk_id": f"para_{chunk_counter}",
                        "strategy": "paragraph",
                        "filename": page["filename"],
                        "page_number": page["page_number"],
                        "text": chunk_text,
                        "word_count": current_word_count
                    })

                    # Paragraph overlap: retain last paragraph if its length is <= overlap_words * 1.5
                    last_para = current_paras[-1]
                    last_para_wc = len(last_para.split())
                    if last_para_wc <= overlap_words * 1.5:
                        current_paras = [last_para, para]
                        current_word_count = last_para_wc + para_words
                    else:
                        current_paras = [para]
                        current_word_count = para_words
                else:
                    current_paras.append(para)
                    current_word_count += para_words

            # Trailing paragraphs for this page
            if current_paras:
                chunk_counter += 1
                chunk_text = "\n\n".join(current_paras)
                chunks.append({
                    "chunk_id": f"para_{chunk_counter}",
                    "strategy": "paragraph",
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "text": chunk_text,
                    "word_count": current_word_count
                })

        return chunks


class VectorStoreManager:
    """
    Manages SentenceTransformers embedding generation, normalized FAISS IndexFlatIP indexing,
    and index persistence.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        vector_store_dir: str = "vector_store"
    ):
        self.model_name = model_name
        self.vector_store_dir = vector_store_dir
        os.makedirs(self.vector_store_dir, exist_ok=True)
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loads SentenceTransformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_and_save_index(
        self,
        chunks: List[Dict[str, Any]],
        strategy: str
    ) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
        """
        Embeds chunks with L2 normalization, creates FAISS IndexFlatIP, and saves to disk.
        """
        if not chunks:
            raise ValueError("No chunks provided to build vector index.")

        texts = [chunk["text"] for chunk in chunks]
        # Normalize embeddings so that Inner Product == Cosine Similarity
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # File paths
        index_file = os.path.join(self.vector_store_dir, f"index_{strategy}.faiss")
        meta_file = os.path.join(self.vector_store_dir, f"metadata_{strategy}.json")

        faiss.write_index(index, index_file)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return index, chunks

    def load_index(
        self,
        strategy: str
    ) -> Optional[Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]]:
        """
        Loads index and metadata for the given chunking strategy from disk if available.
        """
        index_file = os.path.join(self.vector_store_dir, f"index_{strategy}.faiss")
        meta_file = os.path.join(self.vector_store_dir, f"metadata_{strategy}.json")

        if os.path.exists(index_file) and os.path.exists(meta_file):
            try:
                index = faiss.read_index(index_file)
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                return index, metadata
            except Exception:
                return None
        return None


class Retriever:
    """
    Performs cosine similarity search using normalized FAISS IndexFlatIP.
    Returns top-k candidate passages with similarity scores and formatted source IDs.
    """

    def __init__(
        self,
        vector_store_mgr: VectorStoreManager,
        index: faiss.IndexFlatIP,
        chunks: List[Dict[str, Any]],
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ):
        self.vector_store_mgr = vector_store_mgr
        self.index = index
        self.chunks = chunks
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant passages for query.
        Returns list of passages with keys:
          - source_id: e.g. '[S1]', '[S2]'
          - rank: 1-indexed
          - score: float cosine similarity
          - filename: source PDF name
          - page_number: 1-based page
          - text: chunk text
          - meets_threshold: boolean
        """
        if not query.strip() or self.index.ntotal == 0:
            return []

        # Encode query with normalization
        query_embedding = self.vector_store_mgr.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            sim_score = float(score)
            results.append({
                "source_id": f"[S{rank}]",
                "rank": rank,
                "score": round(sim_score, 4),
                "filename": chunk["filename"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "word_count": chunk.get("word_count", len(chunk["text"].split())),
                "meets_threshold": sim_score >= self.similarity_threshold
            })

        return results


class OllamaClient:
    """
    Interfaces with local Ollama server running llama3.2.
    Handles health checks, RAG prompt generation, response parsing, and citation mapping.
    """

    ABSTAIN_RESPONSE = "I don't know based on the provided documents."

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 90
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_connection(self) -> Tuple[bool, str]:
        """
        Checks if Ollama service is running and whether the required model is pulled.
        Returns: (is_ready, message)
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code != 200:
                return False, f"Ollama responded with HTTP status {resp.status_code}."
            
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Match model name with or without tag (e.g. 'llama3.2' or 'llama3.2:latest')
            has_model = any(
                m == self.model_name or m.startswith(f"{self.model_name}:")
                for m in models
            )

            if not has_model:
                return False, (
                    f"Ollama is running, but model '{self.model_name}' is not downloaded. "
                    f"Run `ollama pull {self.model_name}` to download it."
                )

            return True, f"Ollama connected with model '{self.model_name}' available."
        except requests.exceptions.ConnectionError:
            return False, (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure Ollama is installed and running (`ollama serve`)."
            )
        except Exception as e:
            return False, f"Ollama connection error: {str(e)}"

    def build_rag_prompt(self, question: str, passages: List[Dict[str, Any]]) -> str:
        """
        Constructs a strict grounding prompt instructing the model to answer only
        from the supplied passages and include exact inline citations [S1], [S2].
        """
        passages_text = []
        for p in passages:
            passages_text.append(
                f"{p['source_id']} (Document: {p['filename']}, Page: {p['page_number']}):\n{p['text']}"
            )

        context = "\n\n".join(passages_text)

        prompt = f"""You are a truthful, precise Document Q&A assistant.
Answer the user's question STRICTLY and ONLY using the provided source passages below.

CRITICAL RULES:
1. Every factual statement in your answer MUST cite its supporting source using inline brackets like [S1] or [S2].
2. Place citations immediately after the facts they support (e.g., "...according to the framework [S1].").
3. Do NOT invent information, facts, page numbers, or citation IDs not present in the passages.
4. If the provided passages do not contain sufficient evidence to answer the question, you MUST respond EXACTLY with:
{self.ABSTAIN_RESPONSE}
5. Do NOT include prior knowledge or speculation outside the passages.
6. Only cite source IDs that are provided in the Sources list below.

Provided Sources:
{context}

Question: {question}

Answer:"""
        return prompt

    def generate_answer(
        self,
        question: str,
        retrieved_passages: List[Dict[str, Any]],
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Executes question answering:
        - Validates evidence threshold
        - Calls local Ollama llama3.2
        - Extracts and verifies inline citations
        - Maps citations to source passages
        """
        # 1. Relevance check: if no passages meet the threshold, abstain immediately
        qualifying_passages = [p for p in retrieved_passages if p["score"] >= similarity_threshold]

        if not qualifying_passages:
            return {
                "answer": self.ABSTAIN_RESPONSE,
                "is_abstained": True,
                "abstain_reason": (
                    f"No retrieved passages met the relevance threshold of {similarity_threshold:.2f}. "
                    f"Highest similarity score was {retrieved_passages[0]['score'] if retrieved_passages else 0.00:.2f}."
                ),
                "citations": [],
                "raw_response": self.ABSTAIN_RESPONSE,
                "passages": retrieved_passages
            }

        # 2. Check Ollama connectivity
        is_ready, conn_msg = self.check_connection()
        if not is_ready:
            raise ConnectionError(conn_msg)

        # 3. Build prompt and query Ollama
        prompt = self.build_rag_prompt(question, qualifying_passages)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Zero temperature for deterministic, factual output
                "top_p": 0.9
            }
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama API returned HTTP {resp.status_code}: {resp.text}")

            result_json = resp.json()
            raw_answer = result_json.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with Ollama: {str(e)}")

        # 4. Check for model abstention
        cleaned_answer = raw_answer.strip()
        if (
            self.ABSTAIN_RESPONSE.lower() in cleaned_answer.lower()
            or "i do not know based on the provided documents" in cleaned_answer.lower()
            or "insufficient evidence" in cleaned_answer.lower()
        ):
            return {
                "answer": self.ABSTAIN_RESPONSE,
                "is_abstained": True,
                "abstain_reason": "Model indicated insufficient evidence in retrieved passages.",
                "citations": [],
                "raw_response": raw_answer,
                "passages": retrieved_passages
            }

        # 5. Extract inline citations like [S1], [S2]
        citation_tags = re.findall(r"\[S\d+\]", cleaned_answer)
        unique_tags = sorted(list(set(citation_tags)), key=lambda x: int(x[2:-1]))

        # Map to retrieved passages
        source_map = {p["source_id"]: p for p in qualifying_passages}
        validated_citations = []
        invalid_citations = []

        for tag in unique_tags:
            if tag in source_map:
                validated_citations.append(source_map[tag])
            else:
                invalid_citations.append(tag)

        return {
            "answer": cleaned_answer,
            "is_abstained": False,
            "abstain_reason": None,
            "citations": validated_citations,
            "invalid_citations": invalid_citations,
            "raw_response": raw_answer,
            "passages": retrieved_passages
        }
