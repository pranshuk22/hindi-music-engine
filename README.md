# 🎧 Hindi Music Similarity Engine

An AI-powered music recommendation system built specifically for the acoustic nuances of Hindi, Bollywood, Ghazal, Sufi, and Indian independent music. It outperforms generalist platforms by matching tracks on deep neural embeddings, Indian-specific vocal ornaments, microtonal pitch profiles, and rhythmic texture — rather than relying on global click metrics or Western-centric audio models.

---

## ✨ Features

* **Zero Audio Storage:** Ephemeral pipeline downloads, processes, and immediately deletes all raw audio. Disk footprint stays under 500MB even for 50,000+ songs.
* **Smart Intro Skipping:** Automatically starts the 90-second analysis window at 15s (configurable per song via `clip_start`) to skip instrumental intros common in Bollywood productions.
* **Source Separation First:** Uses `demucs` (htdemucs\_light) to split audio into vocal and instrumental stems before feature extraction, enabling India-specific vocal and rhythmic analysis.
* **Hybrid Feature Vector:** Combines CLAP neural embeddings (512d), multilingual NLP lyric embeddings (384d), and ~95d of handcrafted Indian-specific acoustic features — fused, weighted, and PCA-reduced to 256d before indexing.
* **Indian-Specific Acoustics:** Extracts features designed for Hindustani and Bollywood music topology — microtonal pitch profiles (36-bin PCP replacing standard 12-bin chroma), meend (pitch glide) variance, murki ornamentation index, tabla vs. 808 onset skewness, and vocal energy ratio.
* **Blazing Fast Search:** FAISS (IndexFlatIP for pilot, IndexIVFPQ for scale) queries 50,000+ songs in milliseconds.
* **Lyrics-Aware Matching:** SentenceTransformer embeddings on Hindi/Urdu lyrics prevent acoustic matches that clash in mood or lyric meaning.
* **Interactive UI:** Streamlit app for exploring matches, playing YouTube previews, and refining results via Tick/Cross feedback.

---

## 📂 Project Structure

```text
hindi-music-engine/
│
├── app.py                          # Streamlit web interface
├── requirements.txt                # All Python dependencies
├── run_all.sh                      # One-command pipeline + index build
│
├── data/
│   ├── songs.csv                   # Master song list (title, artist, category, youtube_url, clip_start)
│   ├── features/                   # Handcrafted acoustic features — one JSON per song
│   ├── embeddings/                 # Combined fused vectors — one NPY per song (256d post-PCA)
│   ├── nlp/                        # Lyric embedding vectors — one NPY per song (384d)
│   └── metadata.db                 # SQLite database (WAL mode)
│
├── index/
│   ├── build_index.py              # FAISS vector index builder (also runs PCA)
│   ├── search.py                   # Similarity search and Tick/Cross query engine
│   └── faiss_index.bin             # Saved FAISS index (generated, not committed)
│
├── pipeline/
│   ├── downloader.py               # yt-dlp audio fetching
│   ├── trimmer.py                  # ffmpeg/pydub 90s clipping (configurable start)
│   ├── stemmer.py                  # demucs source separation → vocals + instrumental stems
│   ├── feature_extractor.py        # librosa handcrafted features (MFCC, Chroma, Tempo, HNR, etc.)
│   ├── indian_features.py          # India-specific features (microtonal PCP, meend, murki, onset skewness)
│   ├── embedder.py                 # CLAP embedding + weighted feature fusion + L2 normalisation
│   ├── lyrics_extractor.py         # Genius API lyrics fetcher (with Hindi/Urdu fallback handling)
│   ├── nlp_embedder.py             # SentenceTransformer lyric vectoriser (multilingual)
│   └── processor.py                # Master orchestrator — runs the full pipeline per song
│
├── scripts/
│   ├── run_pipeline.py             # Batch runner (parallel, resumable, with failure logging)
│   ├── test_single.py              # Single-song pipeline smoke test
│   └── evaluate.py                 # Precision@K evaluation script
│
├── utils/
│   └── db.py                       # SQLite helpers (init, insert, get — with WAL + timeout)
│
├── notebooks/
│   └── visualise_clusters.ipynb    # UMAP cluster visualisation and quality checks
│
├── AGENTS.md                       # System context and directives for AI agents
└── FEATURES.md                     # Acoustic feature architecture specification
```plaintext

---

## ⚙️ Setup & Installation

### Prerequisites

* Python 3.10+
* `ffmpeg` installed system-wide

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

### Install Python dependencies

```bash
git clone https://github.com/your-username/hindi-music-engine.git
cd hindi-music-engine

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### First run

```bash
# 1. Test the full pipeline on a single song
python scripts/test_single.py

# 2. Batch process your song list
python scripts/run_pipeline.py --csv data/songs.csv --workers 2

# 3. Build the FAISS index
python index/build_index.py

# 4. Launch the Streamlit app
streamlit run app.py
```

Or use the convenience script to run steps 2–3 sequentially:

```bash
bash run_all.sh
```

---

## 🗂️ songs.csv Format

```csv
title,artist,category,youtube_url,clip_start
Aaj Jaane Ki Zid Na Karo,Farida Khanum,ghazal,,30
Tum Hi Ho,Arijit Singh,sad_romantic,,40
Kun Faya Kun,A.R. Rahman,sufi,,60
Saturday Saturday,Badshah,party,,15
```

* `youtube_url` — optional. If empty, yt-dlp searches by `{title} {artist}`.
* `clip_start` — seconds to skip before the 90s window. Default: 15. Increase for songs with long instrumental intros.
* `category` — used as a metadata label. Current valid values: `ghazal`, `sad_romantic`, `sufi`, `upbeat`, `party`.

---

## 📊 Pipeline Overview

```
URL / Song Name
      │
      ▼
  yt-dlp download
      │
      ▼
  ffmpeg → 90s clip (starts at clip_start)
      │
      ├──────────────────────┐
      ▼                      ▼
  demucs stem split    Lyrics fetch (Genius API)
  vocals / instr            │
      │                     ▼
      ├── Indian features   NLP embed (indic-bert /
      │   (microtonal PCP,  multilingual SentenceTransformer)
      │    meend, murki,         │
      │    onset skew)           ▼
      │                    nlp/song_id.npy
      ├── Vocal features
      │   (vocal MFCC,
      │    energy ratio)
      │
      ├── CLAP embedding
      │   (msclap, 512d)
      │
      ▼
  Weighted fusion → L2 normalise → ~991d vector
      │
      ▼
  PCA → 256d compressed vector
      │
      ▼
  data/embeddings/song_id.npy   +   data/metadata.db
      │
      ▼
  DELETE all audio (raw, clip, stems)
      │
      ▼
  FAISS IndexFlatIP (pilot) / IndexIVFPQ (scale)
```

---

## 🧪 Evaluation

Run the built-in Precision@K evaluation to measure recommendation quality:

```bash
python scripts/evaluate.py
```

This checks: for each song in your database, what fraction of its top-5 recommendations share the same category? A well-tuned system should score above 0.75 on the pilot dataset.

---

## 🔭 Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Single-song pipeline end-to-end | ✅ Done |
| 1 | 50-song pilot — FAISS + UMAP validation | ✅ Done |
| 1b | Indian-specific features + source separation | 🔄 In Progress |
| 2 | Scale to 5,000 songs on Kaggle | ⏳ Planned |
| 3 | Streamlit UI + Tick/Cross feedback loop | ⏳ Planned |
| 4 | Conversational AI discovery interface | ⏳ Planned |
| 5 | Spotify/YouTube playlist export | ⏳ Planned |
| 6 | Scale to 50,000 songs | ⏳ Planned |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `yt-dlp` | Audio download from YouTube |
| `pydub` + `ffmpeg` | Audio clipping and resampling |
| `demucs` | Source separation (vocals / instrumental) |
| `librosa` | Handcrafted acoustic feature extraction |
| `msclap` | CLAP neural audio embedding |
| `crepe` / `librosa.pyin` | Pitch tracking for microtonal features |
| `sentence-transformers` | Multilingual lyric embedding |
| `lyricsgenius` | Genius API lyrics fetcher |
| `faiss-cpu` | Fast approximate nearest-neighbour search |
| `scikit-learn` | PCA dimensionality reduction |
| `umap-learn` | Cluster visualisation |
| `streamlit` | Web UI |
| `sqlite3` | Metadata storage (built-in) |
