# Document Q&A with Citations Using RAG

An academic-grade, fully local Retrieval-Augmented Generation (RAG) system built with **Python 3.12/3.11**, **Streamlit**, **PyMuPDF**, **Sentence Transformers**, **FAISS CPU**, and **Ollama (`llama3.2`)**.

The application allows users to ask questions over a corpus of PDF documents and receive answers strictly grounded in verified source passages. Every answer provides traceable citations containing:
- PDF filename
- 1-based page number
- Exact supporting passage text
- Cosine similarity score

When documents do not contain sufficient evidence, the application abstains with:
> *"I don't know based on the provided documents."*

No paid API keys are required. All embeddings, vector indexing, retrieval, and generation run 100% locally.

---

## 1. Architecture and Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                       PDF Corpus                            │
│                  data/documents/*.pdf                       │
│              (Must have >= 50 total pages)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    PyMuPDF Text Ingestion                   │
│         - Extracts text page-by-page                        │
│         - Preserves filename & 1-based page number          │
│         - Detects scanned PDFs (OCR warning)                │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│  Strategy A: Fixed-Size      │     │  Strategy B: Paragraph-Aware │
│  - ~400 words per chunk      │     │  - Natural paragraph splits  │
│  - ~80 words sliding overlap │     │  - Bundled up to ~400 words  │
│  - Page boundaries preserved │     │  - Paragraph overlap         │
└──────────────┬───────────────┘     └──────────────┬───────────────┘
               │                                    │
               └──────────────────┬─────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│          Sentence Transformers (all-MiniLM-L6-v2)           │
│        - Normalizes L2 vectors for cosine similarity        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FAISS CPU Vector Store                   │
│            - IndexFlatIP with normalized vectors            │
│            - Fast, exact cosine similarity retrieval        │
│            - Persisted to vector_store/                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                Retriever & Relevance Gate                   │
│         - Retrieves Top 5 candidate passages                │
│         - Filters by similarity threshold (default: 0.35)   │
│         - Abstains if max similarity < threshold            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                Ollama LLM (llama3.2 Local)                  │
│       - Grounded prompt enforcing inline [S#] citations     │
│       - Strict fallback instruction if evidence is weak     │
│       - Post-validation: checks citations against sources   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Web Interface                   │
│     - Inline citation badges ([S1], [S2])                   │
│     - Expandable verified source passage inspector          │
│     - Strategy switching & real-time corpus metrics         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Project Folder Structure

```
document-qa/
├── app.py                      # Streamlit interactive web interface
├── evaluate.py                 # Evaluation benchmark script (Strategy A vs B)
├── rag_pipeline.py             # Core modular RAG pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Full documentation & usage guide
├── data/
│   └── documents/              # Input PDF corpus (place >= 50 pages here)
│       ├── NIST.AI.100-1.pdf
│       └── NIST.AI.600-1.pdf
├── vector_store/               # Persisted FAISS indices and chunk metadata
│   ├── index_fixed.faiss
│   ├── metadata_fixed.json
│   ├── index_paragraph.faiss
│   └── metadata_paragraph.json
└── results/
    └── chunking_comparison.csv # Actual empirical evaluation results
```

---

## 3. Installation and Setup (Windows PowerShell)

### Step 1: Clone / Navigate to Project Root
```powershell
cd "d:\Projects\Virtual Internship\Document-QA-With-Citations"
```

### Step 2: Create and Activate Virtual Environment
Using Python 3.11 or 3.12:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Install and Start Ollama
1. Download Ollama for Windows from [ollama.com](https://ollama.com/download) or install via winget:
   ```powershell
   winget install Ollama.Ollama
   ```
2. Start the Ollama background service:
   ```powershell
   ollama serve
   ```
3. Download the required model (`llama3.2`):
   ```powershell
   ollama pull llama3.2
   ```

---

## 4. Adding PDF Documents

Place your PDF documents in:
```
data/documents/
```

### Corpus Constraints:
- **Total Page Count:** The pipeline requires **at least 50 total pages** across all PDFs in `data/documents/`. If fewer than 50 pages are present, the system will stop with an informative error.
- **Text Layer Required:** The PDFs must contain extractable text. If an image-only or scanned PDF is detected (0 extractable text characters), the system flags an OCR warning.

*Included Sample Documents:*
The repository includes two comprehensive documents from the National Institute of Standards and Technology (NIST):
1. `NIST.AI.100-1.pdf` — NIST AI Risk Management Framework (AI RMF 1.0)
2. `NIST.AI.600-1.pdf` — Generative AI Profile

Together, these documents contain over 140 pages, satisfying the 50-page minimum requirement.

---

## 5. Running the Streamlit Application

Start the Streamlit application:
```powershell
streamlit run app.py
```

The browser will open at `http://localhost:8501`.

### Features of the Web Interface:
- **Corpus & Index Metrics:** Real-time metrics showing total PDF pages, extracted pages, total chunks for the active strategy, and total documents.
- **Strategy Selector:** Switch seamlessly between **Fixed-size (Strategy A)** and **Paragraph-aware (Strategy B)**. Switching strategies dynamically updates chunks and loads/builds the respective FAISS index.
- **Similarity Threshold Slider:** Adjust cosine similarity gate (default: `0.35`). Passages below this score are filtered out.
- **Rebuild Index Button:** Forces re-extraction and re-indexing when documents are added or modified.
- **Ask Question & Sample Prompts:** Ask custom queries or select from pre-configured sample questions.
- **Verified Source Citations:** View inline citations (`[S1]`, `[S2]`) with collapsible inspection cards displaying the document name, page number, cosine similarity score, and exact text passage.
- **Ranked Candidate Explorer:** View all top-5 retrieved passages and their similarity metrics.

---

## 6. Running the Evaluation Benchmark

Run the comparative evaluation experiment:
```powershell
python evaluate.py
```

### Evaluation Metrics:
1. **Recall@5:** Proportion of answerable test queries where at least one ground-truth passage was retrieved among the top 5 candidates.
2. **Mean Reciprocal Rank (MRR):** Measures where the first relevant passage appears in the retrieved ranking ($1 / \text{rank}$).
3. **Average Retrieval Latency (ms):** Average time taken to encode the query and retrieve top-5 passages from FAISS.
4. **Number of Chunks:** Total chunks generated by the strategy across the corpus.
5. **Abstention Accuracy:** Percentage of out-of-domain / unanswerable questions correctly rejected by the similarity gate.

### Evaluation Results:
Results are saved to:
```
results/chunking_comparison.csv
```

---

## 7. Chunking Strategies Compared

### Strategy A: Fixed-Size Word Chunking
- **Target Size:** ~400 words per chunk.
- **Overlap:** ~80 words sliding window overlap.
- **Boundary Preservation:** Preserves page boundaries strictly. Chunks do not cross pages, ensuring citations unambiguously point to the exact source page.
- **Characteristics:** Consistent chunk sizes; high recall for queries where information is evenly distributed. May occasionally bisect a paragraph or sentence.

### Strategy B: Paragraph-Aware Chunking
- **Boundary Splitting:** Splits on paragraph breaks (`\n\n+`).
- **Combining:** Aggregates paragraphs until ~400 words is reached.
- **Paragraph Overlap:** Preserves previous paragraph overlap when moving to the next chunk.
- **Characteristics:** Respects semantic integrity; answers retain natural context and complete thoughts.

---

## 8. Citation Grounding and Abstention Behavior

### Citation Verification
- When Ollama answers, it is strictly prompted to include inline citations (`[S1]`, `[S2]`).
- The application parses every citation tag and validates it against the retrieved set.
- If the model invents a citation ID (e.g. `[S9]` when only 5 passages exist), it is flagged as invalid.
- Each citation references the exact file and 1-based page number preserved from PyMuPDF.

### Honest Abstention
A similarity threshold alone does not guarantee answer correctness. However, it filters out queries with no semantic overlap. The system abstains under three conditions:
1. **Retrieval Threshold Abstention:** If the highest passage cosine similarity score is below the configured threshold (e.g. `< 0.35`), the application abstains before querying the LLM:
   > *"I don't know based on the provided documents."*
2. **Model Abstention:** If retrieved passages do not contain sufficient evidence to answer the user's specific question, the model responds with the fallback phrase.
3. **Empty Corpus / Retrieval Fail:** If no documents or passages match.

---

## 9. Troubleshooting & FAQ

| Issue | Cause | Solution |
|---|---|---|
| `Corpus page requirement not met (< 50 pages)` | `data/documents/` contains fewer than 50 pages | Place additional PDF files in `data/documents/` until total pages exceed 50. |
| `Cannot connect to Ollama at http://localhost:11434` | Ollama service is not running | Run `ollama serve` in a separate terminal. |
| `Model 'llama3.2' is not downloaded` | Model has not been pulled | Run `ollama pull llama3.2`. |
| `OCR Warning: 0 extractable text characters` | Document is scanned or image-only | Run an OCR tool (e.g., Tesseract or Adobe Acrobat OCR) to create a searchable text layer. |
| `ModuleNotFoundError` | Virtual environment not active or dependencies missing | Run `.\.venv\Scripts\Activate.ps1` and `pip install -r requirements.txt`. |
| Index not updating after PDF changes | Using cached vector index | Click the **🔄 Rebuild / Load Index** button in the Streamlit sidebar. |
