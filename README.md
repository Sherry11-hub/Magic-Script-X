# 🎤 Generative AI Event & Script Copilot

> An intelligent RAG-powered assistant for festival and conference organizers —
> answers logistical questions, drafts anchoring scripts, and adapts schedules
> in real time.

---

## ✨ Features

| Feature | Description |
|---|---|
| **📂 Data Ingestion Hub** | Upload `.txt`, `.pdf`, `.csv` event documents. Auto-chunks, embeds, and stores in ChromaDB. |
| **🎤 Script Copilot** | Chat interface for drafting anchoring scripts, speaker intros, transitions, and closing ceremonies — grounded in your documents. |
| **📅 Schedule Adapter** | Report a disruption in plain English; get a fully revised, cascaded schedule instantly. |
| **🔒 Anti-hallucination** | Both AI pipelines are explicitly instructed to use *only* the uploaded context. |
| **🌊 Streaming Output** | Token-by-token response streaming via `st.write_stream()`. |
| **📥 Export** | Download generated scripts and revised schedules as `.txt` files. |

---

## 🏗️ Architecture

```
event-copilot/
├── app.py                     ← Streamlit entry point & orchestrator
├── config.py                  ← All settings loaded from .env
├── requirements.txt
├── .env.example               ← Copy to .env and fill in credentials
│
├── core/
│   ├── document_processor.py  ← Load → Chunk → Embed → Store pipeline
│   └── rag_engine.py          ← Retrieve → Prompt → Generate pipelines
│
├── ui/
│   ├── sidebar.py             ← Data Ingestion Hub (sidebar)
│   ├── script_copilot.py      ← Script Copilot tab
│   └── schedule_adapter.py    ← Schedule Adapter tab
│
└── sample_data/               ← Demo documents to test immediately
    ├── techwave_schedule.txt
    └── speaker_bios.txt
```

### Data Flow

```
User uploads file
       │
       ▼
document_processor.py
  load_document_from_upload()   ← Temp file → LangChain loader
       │
  chunk_documents()             ← RecursiveCharacterTextSplitter
       │
  get_embedding_function()      ← HuggingFace or OpenAI embeddings
       │
  ChromaDB.add_documents()      ← Persisted to ./chroma_db/
       │
       ▼
User submits query
       │
       ▼
rag_engine.py
  _get_retriever(k=N)           ← MMR retrieval from ChromaDB
       │
  _format_docs()                ← Context string with source headers
       │
  ChatPromptTemplate            ← Anti-hallucination system prompt
       │
  ChatOpenAI (streaming=True)   ← GPT-3.5/GPT-4 via OpenAI API
       │
  StrOutputParser               ← Token stream
       │
       ▼
st.write_stream()               ← Live display in Streamlit
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone & install

```bash
git clone https://github.com/your-repo/event-copilot.git
cd event-copilot

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **First run note:** HuggingFace will download the `all-MiniLM-L6-v2` model
> (~90 MB) to `./.model_cache/`. This is a one-time download.

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
OPENAI_API_KEY=sk-your-key-here
```

All other values have sensible defaults.

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Try the sample data

1. In the sidebar, click **Browse files** and upload both files from `sample_data/`:
   - `techwave_schedule.txt`
   - `speaker_bios.txt`
2. Click **🚀 Process & Index Documents**
3. Switch to the **Script Copilot** tab and try:
   > *"Write a professional 60-second introduction for the morning keynote speaker."*
4. Switch to the **Schedule Adapter** tab and try:
   > *"The 2:00 PM RAG workshop is delayed by 30 minutes. Revise the afternoon schedule."*

---

## ⚙️ Configuration Reference

All settings are in `.env`. See `.env.example` for full descriptions.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI secret key |
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI model to use |
| `LLM_TEMPERATURE` | `0.3` | Creativity (0=deterministic, 1=creative) |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` or `openai` |
| `HF_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model name |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RETRIEVAL_K` | `5` | Chunks retrieved per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB stores data |

### Switching to OpenAI Embeddings

```env
EMBEDDING_PROVIDER=openai
# Uses text-embedding-3-small automatically (better quality, costs ~$0.02/1M tokens)
```

### Upgrading to GPT-4

```env
LLM_MODEL=gpt-4o
MAX_TOKENS=2000
```

---

## 📐 Technical Design Decisions

### Chunking Strategy
`RecursiveCharacterTextSplitter` with `chunk_size=800, overlap=150` and
separators in priority order: `["\n\n", "\n", ". ", " ", ""]`.

- **Why recursive?** Preserves semantic boundaries (paragraphs > sentences > words)
  before falling back to hard character cuts.
- **Why 800 chars?** Balances retrieval precision with enough context for the LLM.
  Dense schedules with many sessions work better with 600–800 char chunks.
- **Why 150 overlap?** ~18% overlap prevents answers from falling across boundaries.

### Retrieval Strategy
MMR (Maximal Marginal Relevance) with `lambda_mult=0.6`:

- Avoids retrieving N near-duplicate chunks (e.g., same table repeated on every page).
- Balances relevance (0.6 weight) vs. diversity (0.4 weight).
- Schedule Adapter uses `k=8` (vs. `k=5` for Script Copilot) to capture the full schedule.

### Anti-Hallucination
Both system prompts contain an explicit instruction:

> *"Answer using ONLY the provided context. If the answer is not in the context,
> state that you do not have enough information."*

The LLM is given context chunks tagged with their source file and page number,
making it harder to confabulate.

---

## 🔧 Extending the App

### Add a new document type
In `core/document_processor.py`, extend the `load_document_from_upload()` function:

```python
elif ext == ".docx":
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(tmp_path)
    docs = loader.load()
```

### Add a new AI feature tab
1. Create `ui/my_feature.py` with a `render_my_feature()` function.
2. Import and add a new tab in `app.py`:

```python
from ui.my_feature import render_my_feature

tab_script, tab_schedule, tab_new = st.tabs([...])
with tab_new:
    render_my_feature()
```

3. Add a pipeline function in `core/rag_engine.py` following the
   `run_script_copilot()` pattern.

### Use Pinecone instead of ChromaDB
```python
# In core/document_processor.py
from langchain_pinecone import PineconeVectorStore
store = PineconeVectorStore(index_name="event-copilot", embedding=embedding_fn)
```

---

## 🛟 Troubleshooting

| Problem | Solution |
|---|---|
| `OPENAI_API_KEY not set` | Copy `.env.example` → `.env` and add your key |
| `Knowledge base is empty` | Upload documents and click "Process & Index Documents" |
| Slow first startup | HuggingFace is downloading the embedding model (~90 MB) — normal |
| Duplicate content after re-upload | `Chroma.add_documents()` is additive by design. Use "Clear KB" to reset |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in your virtual env |
| PDF not loading | Ensure `pypdf` is installed: `pip install pypdf` |

---

## 📄 License

MIT License — see `LICENSE` for details.
