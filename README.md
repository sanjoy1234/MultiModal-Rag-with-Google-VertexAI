# 🪄 GCP Multi-Modal RAG with Vertex AI + AstraDB (Cassandra) + LangChain

A reference implementation that shows how to wire up **Google Vertex AI’s Gemini model**, **DataStax AstraDB (serverless Cassandra-with-vectors)**, and **LangChain** to build a **production-grade, multi-modal Retrieval-Augmented Generation (RAG) service**—able to reason over both **text *and* images** at enterprise scale.

---

## ✨ Key Capabilities

| Capability | What It Does |
|------------|--------------|
| **Vertex AI Gemini 1.0 Pro / Flash** | Large-context LLM that supports streaming, text-to-text, and image-to-text prompts. |
| **AstraDB Vector Store** | Serverless Cassandra table that stores → retrieves vector embeddings (up to 8 K dims). |
| **LangChain Orchestration** | Chains, agents, and prompt-templates that glue the LLM + vector store into a unified RAG workflow. |
| **Multi-Modal Support** | Generates image embeddings (using Vertex AI multimodal embeddings) and stores them alongside text chunks. |
| **Streaming & Tuning Hooks** | Demonstrates how to stream responses and tweak generation parameters (temperature, top‑p, etc.). |

---

## 🏗️ Architecture

```
┌────────────┐     text / image      ┌──────────────┐   embed + store   ┌──────────────┐
│  User / UI │ ───────────────────▶ │  LangChain    │ ────────────────▶ │  AstraDB      │
└────────────┘   query + context     │  (chains +   │                   │  (vectors)    │
       ▲                             │    agents)   │ ◀───────────────┐ │              │
       │            JSON answer      └──────────────┘   nearest‑N      │              │
       │                                                   vectors     └──────────────┘
       │                                                               ▲
       │                         prompt + retrieved docs               │
       └────────────────────── Vertex AI Gemini  ──────────────────────┘
```

---

## ⚙️ Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Tested on 3.10 & 3.11 |
| **GCP project with Vertex AI enabled** | Billing must be on & you need access to the *Gemini 1.0 Pro* and *Multimodal Embeddings* models. |
| **DataStax AstraDB instance** | Create a free‑tier database & generate an application token. |
| **Service‑account JSON key** | Recommended for non‑Colab environments. |

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/gcp-multimodal-rag-vertex-ai-astradb.git
cd gcp-multimodal-rag-vertex-ai-astradb

# 2. Create & activate a virtualenv
python -m venv .venv
source .venv/bin/activate

# 3. Install deps
pip install --upgrade google-cloud-aiplatform                        ragstack-ai                        langchain                        langchain-google-vertexai==2.0.5                        cassandra-driver                        python-dotenv
```

---

## 🔐 Environment Variables

Create a `.env` file or export the vars in your shell:

```dotenv
# GCP
GCP_PROJECT_ID=your-gcp-project
LOCATION=us-central1           # or europe-west4, asia-southeast1, …

# Vertex AI models
VERTEX_TEXT_MODEL=gemini-1.0-pro
VERTEX_IMAGE_EMBED_MODEL=multimodalembedding@001

# AstraDB
ASTRA_DB_API_ENDPOINT=https://<db-id>-<region>.apps.astra.datastax.com
ASTRA_DB_APPLICATION_TOKEN=AstraCS:…      # NEVER hard‑code in code!
ASTRA_NAMESPACE=vertex_rag_demo
ASTRA_COLLECTION=multimodal_docs
```

> **Heads‑up:**  
> The sample notebook in the repo contains placeholder tokens that must be replaced with *your* secrets or, better, loaded from environment variables.

---

## 🚀 Running the Demo

```bash
python gcp_multimodal_rag_using_vertex_ai_astradb-langchain.py
```

The script will:

1. **Authenticate** to GCP (service‑account or `gcloud auth login`) and initialise Vertex AI.  
2. **Connect** to AstraDB and create the vector collection if it doesn’t exist.  
3. **Ingest** a sample set of images + captions into the vector store (both text & image embeddings).  
4. **Build** a LangChain `RetrievalQA` chain that:  
   * embeds the incoming (text or image) query  
   * fetches the top‑`k` nearest neighbours from AstraDB  
   * composes a prompt with the retrieved context  
   * calls Gemini in **streaming** mode for low‑latency partial results  
5. **Print** the model’s answer and metadata (source docs, generation time, token counts).

---

## 🧑‍💻 Quick‑Start Example

```python
from chains import multimodal_rag_chain

question = "📸  What landmark is in this picture and why is it architecturally significant?"
answer = multimodal_rag_chain(
    query_image_path="assets/golden_gate.jpg",
    query_text=None,
    stream=True,
)

print("\n\n🤖", answer)
```

---

## 🛠️ Customization Tips

| What to Change | How |
|----------------|-----|
| **Model variant** | Swap `gemini-1.0-pro` for `gemini-1.0-flash` in `VERTEX_TEXT_MODEL`. |
| **Temperature / Top‑p** | Edit the `GenerationConfig` block. |
| **Vector distance** | Adjust `similarity_metric="cosine"` vs `"dot_product"` in the AstraDB collection. |
| **Hybrid Search** | Combine vector + keyword search by adding Astra’s `filter` clause in the retriever. |
| **Chunking strategy** | Replace the default `RecursiveCharacterTextSplitter` in `ingest.py`. |

---

## 📂 Repo Layout

```
.
├─ ingest.py                # one‑off loader to push docs/images into AstraDB
├─ gcp_multimodal_rag_using_vertex_ai_astradb-langchain.py  # main demo script
├─ chains.py                # reusable LangChain chains / prompts
├─ requirements.txt
└─ README.md                # ← you are here
```

---

## ✅ Roadmap

- [ ] Add **Cloud Run** deployment guide  
- [ ] Example **Dockerfile** with gcloud + cassandra-driver baked in  
- [ ] **FastAPI** wrapper for easy REST / gRPC serving  
- [ ] Optional **Cloud Storage** bucket for raw image assets  
- [ ] **CI/CD** GitHub Actions workflow with unit tests

---

## 🤝 Contributing

PRs are welcome! Please open an issue first to discuss major changes.  
Make sure to **remove or redact any secrets** before committing.

---

## 📝 License

[MIT](LICENSE)

---

### ⭐ If you find this useful, give the repo a star and drop me a note on LinkedIn!
