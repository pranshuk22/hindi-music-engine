# 🤖 System Context & AI Agent Directives (AGENTS.md)

## 📌 Project Overview

**Name:** Hindi Music Similarity Engine & Discovery App
**Goal:** Build a highly accurate, domain-specific music recommendation product optimised for Hindi and Indian regional music. The system features an interactive UI for playlist generation, vibe-based discovery, and real-time preference learning — powered by Indian-specific acoustic embeddings rather than global click metrics.

**Primary constraint:** Free-tier compute only. No paid APIs, no paid GPU. All models must run locally or on Kaggle/Colab free tier. All third-party services must have a usable free tier.

---

## 🚀 Core Product Features

1. **Song Similarity Search:** Given a song (URL or name), return the top-K acoustically similar songs from the database, ranked by cosine similarity in the fused 256d embedding space.
2. **Playlist Expansion Engine:** Users input a Spotify or YouTube playlist link. The system extracts the tracks, computes a Playlist Centroid Vector (mean of all track embeddings), and queries FAISS to surface similar undiscovered songs.
3. **Conversational AI Discovery:** A chat interface where users describe their desired vibe (e.g., "songs like Kabira but faster," "something melancholic like old Jagjit Singh"). An LLM translates the natural language into structured vector queries and metadata filters.
4. **Active Preference Learning (Tick/Cross Loop):** Real-time feedback mechanism. When a user rejects (❌) a recommendation, its vector is subtracted from the query vector (Rocchio-style). When a user approves (✅), its vector is added. The updated query vector is immediately re-queried against FAISS — no model retraining required.
5. **Playlist Export:** Users save generated playlists directly to their Spotify or YouTube accounts via OAuth2.

---

## 🛠 System Architecture

The system is split into three distinct layers. Each layer is independently deployable and testable.

### Layer 1 — Ingestion Pipeline (Offline / Batch)

Runs offline, on Kaggle, or triggered by the async worker queue. Never runs synchronously on the user-facing request thread.

* **Fetch:** `yt-dlp` downloads audio. Searches by `{title} {artist}` if no URL is provided.
* **Trim:** `pydub`/`ffmpeg` extracts a 90-second clip. Start time is configurable per song via `clip_start` in `songs.csv` (default: 15s).
* **Stem:** `demucs` (htdemucs\_light checkpoint) separates the clip into `clip_vocals.wav` and `clip_instr.wav`.
* **Extract:** `librosa` + `pipeline/indian_features.py` compute the full handcrafted feature set (~95 dims). See `FEATURES.md` for the complete schema.
* **Embed:** `msclap` generates a 512d CLAP embedding. `SentenceTransformers` generates a 384d lyric embedding (if lyrics available).
* **Fuse:** All feature groups are independently L2-normalised, then concatenated with explicit weights before a final L2 normalisation. See `FEATURES.md` — Fusion & Weighting section.
* **Compress:** PCA reduces the fused vector to 256d. The PCA model is fit once on the pilot dataset and reused for all subsequent songs.
* **Store:** `data/embeddings/song_id.npy` (256d vector), `data/features/song_id.json` (raw features), `data/nlp/song_id.npy` (384d lyric vector), `data/metadata.db` (SQLite).
* **Cleanup:** All audio files (raw download, 90s clip, stems) are deleted immediately after extraction.

### Layer 2 — Retrieval & API Backend (Real-Time)

* **Vector Store:** `faiss-cpu` — `IndexFlatIP` for pilot (<1,000 songs), `IndexIVFPQ` for scale (>10,000 songs).
* **Search:** Cosine similarity via inner product on L2-normalised vectors.
* **Query Engine:** Handles three query types — single song, playlist centroid, and adjusted Rocchio query vector.
* **LLM Orchestrator:** Parses conversational prompts into structured filters. **Use Groq API (free tier, Llama 3.1 70B)** for cloud inference, or **Ollama with Llama 3.2 3B** for fully local inference. Do not use OpenAI or any paid API.
* **Metadata Filter Layer:** Post-FAISS result filtering by `category`, `language`, `tempo_bucket`, or artist. Applied after vector retrieval, not before — filtering in vector space is handled by metadata, not by modifying the query vector.

### Layer 3 — Frontend (User-Facing)

* **Current:** Streamlit (`app.py`) — fast to develop, sufficient for MVP and pilot user testing.
* **Phase 2 target:** Next.js + FastAPI if the Streamlit app proves too limiting (e.g., cannot support real-time websocket feedback loop or proper OAuth flows).
* **Song cards:** Display title, artist, category, similarity score, and an embedded YouTube player.
* **Feedback buttons:** ✅ Tick and ❌ Cross per song card, wired to the Rocchio query engine.

---

## 🤖 Directives for AI Agents Working on this Codebase

These are hard rules. Do not deviate from them.

### 1. Ephemeral Storage — Non-Negotiable

Never modify the ingestion pipeline to retain `.wav` files permanently. The `cleanup()` function in `pipeline/trimmer.py` must be called after every audio file — raw download, 90s clip, and both stems. Violation of this breaks the storage budget at scale.

### 2. Vector Math Over Model Retraining

The Tick/Cross feedback loop **must** use Rocchio-style query vector adjustment:

```
new_query = α * original_query + β * sum(approved_vectors) - γ * sum(rejected_vectors)
```

Use `α=1.0, β=0.75, γ=0.25` as defaults. Do not attempt to retrain CLAP, fine-tune embeddings, or modify the FAISS index in real-time. All feedback is handled as vector algebra in memory during the session.

### 3. DB Schema — Source of Truth is `utils/db.py`

Always refer to `utils/db.py` for the current schema, not `context.txt` or any snapshot file. The canonical columns are:

```
id, title, artist, youtube_url, category, duration, tempo,
embedding_path, features_path, nlp_path, lyrics_missing,
processed, created_at
```

`lyrics_missing` (INTEGER, 0/1) must be set to `1` when Genius API returns no result, so downstream code can zero out the NLP vector correctly rather than using stale or hallucinated data.

### 4. PCA Model Consistency

The PCA model (sklearn `PCA(n_components=256)`) is fit **once** on the pilot dataset embeddings and saved to `index/pca_model.pkl`. All new songs added later — including those added by the async worker — must use `pca_model.transform()`, never `pca_model.fit_transform()`. Refitting on a subset will shift the entire embedding space and invalidate the FAISS index.

### 5. LLM Usage — Free Tier Only

For the conversational discovery feature, use one of the following and no others:

* **Groq API** (free tier) — `llama-3.1-70b-versatile` model. Best for cloud deployment. Handles Hindi transliteration in prompts well.
* **Ollama local** — `llama3.2:3b` model. Best for fully offline use. Lower quality but zero cost and no rate limits.

The LLM's only job is to parse a natural language prompt into a JSON structure:

```json
{
  "anchor_song": "Kabira",
  "anchor_artist": "Tochi Raina",
  "tempo_modifier": "faster",
  "category_filter": null,
  "mood_keywords": ["melancholic", "acoustic"]
}
```

The vector search logic must never be inside the LLM prompt. The LLM is a parser, not a recommender.

### 6. Async Ingestion — Simple Queue First

When a user queries an unknown song, it must not block the UI thread. Use Python's built-in `queue.Queue` with a `threading.Thread` background worker. Do **not** introduce Celery or Redis at this stage. The system does not have the user volume to justify that infrastructure. Upgrade to Celery only when concurrent ingestion requests exceed what a single background thread can drain.

The async ingestion flow:

```
User requests unknown song
        │
        ▼
Background queue receives (title, artist, url)
        │
        ▼
UI immediately returns: "This song isn't in our library yet.
We're processing it — check back in a few minutes."
        │
        ▼
Worker thread runs full pipeline → inserts to DB + updates FAISS index
        │
        ▼
Next user query for same song hits the DB successfully
```

### 7. API-First Design (Phase 2 Transition)

All core logic — FAISS querying, playlist centroid calculation, Rocchio adjustment, LLM prompt parsing — must be implemented as isolated Python functions with clean signatures before being wired into any UI. This ensures the Streamlit → Next.js/FastAPI transition does not require rewriting business logic.

### 8. Parallel Pipeline Safety

`run_pipeline.py` uses `ThreadPoolExecutor`. The CLAP model (`msclap`) must be loaded **once** at module level, not inside the worker function. Loading it per-worker doubles memory usage and will OOM on 8GB RAM machines with 2 workers. The demucs model has the same constraint — load once, share across threads.

---

## 🌟 Phase 2: Consumer Application Roadmap

### Phase 2a — Scaled Database (Month 2–3)

* Scale from 50 pilot songs to 5,000 songs using Kaggle batch processing.
* Kaggle workflow: upload `songs.csv` as a dataset input → notebook runs pipeline → commit output embeddings as a dataset.
* Switch FAISS to `IndexIVFPQ` once song count exceeds 1,000.
* Refit PCA on the larger embedding set and rebuild the FAISS index.

### Phase 2b — Conversational Discovery (Month 3–4)

* Integrate Groq/Ollama LLM for natural language prompt parsing.
* Map parsed JSON to FAISS query + metadata filter.
* Support tempo modifiers ("faster," "slower"), mood keywords, and artist anchoring.

### Phase 2c — Tick/Cross Feedback Loop (Month 4)

* Wire Streamlit song cards with ✅/❌ buttons.
* Implement Rocchio vector adjustment in `index/search.py`.
* Session state holds the current query vector; each button press modifies it in-memory and re-queries FAISS.

### Phase 2d — Playlist Export (Month 5)

* Spotify OAuth2 (`playlist-modify-public` scope) via `spotipy` library.
* YouTube Data API v3 (`playlistItems.insert`) for YouTube export.
* Both are free APIs with generous rate limits for personal/low-volume use.

### Phase 2e — Scale to 50,000 Songs (Month 5–6)

* Multi-session Kaggle processing with chunked CSVs.
* Language metadata column (`language`: Hindi, Punjabi, Urdu, etc.) to enable cross-lingual filtering.
* Migrate SQLite to PostgreSQL if write concurrency becomes a bottleneck.

---

## ⚠️ Known Constraints and Mitigations

| Constraint | Impact | Mitigation |
|---|---|---|
| No GPU on laptop | demucs very slow (~3–5 min/song on CPU) | Use `htdemucs_light` checkpoint; run bulk processing on Kaggle T4 GPU |
| Genius API rate limits | ~1 req/sec, lyrics missing for ~30% of older songs | Exponential backoff; set `lyrics_missing=1` and zero-pad NLP vector |
| CLAP not trained on Indian music | Misclassifies high-energy sufi as party | Compensated by handcrafted Indian-specific features with explicit weighting |
| SQLite under parallel writes | "Database is locked" errors | WAL mode + `timeout=15.0` on all connections |
| 8GB laptop RAM | OOM risk with 2 parallel workers | Load CLAP + demucs once at module level; 2 workers max |