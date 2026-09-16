# 🧮 Acoustic Feature Architecture (FEATURES.md)

## 📌 Directive for AI Agents

When writing extraction code or modifying the vector space, agents MUST prioritise Indian music topology. Do not rely solely on Western 12-tone equal temperament libraries — standard 12-bin chroma, standard beat tracking, and generic MFCC are insufficient for Hindustani and Urdu music.

**All songs undergo source separation before deep feature extraction.** The 90-second clip is split into `clip_vocals.wav` and `clip_instr.wav` using `demucs` (htdemucs\_light checkpoint) before any librosa or Indian feature extraction runs. Stems are deleted immediately after extraction.

**The final FAISS-indexed vector is 256 dimensions**, produced by PCA compression of the ~991d fused vector. The PCA model is fit once on the pilot dataset and frozen. Never refit it on new data.

---

## 🧬 Feature Vector Schema

Total raw dimensions before fusion: **~991d**
Final stored/indexed dimensions after PCA: **256d**

---

### Group 1 — Neural & Semantic Features

| Feature | Dims | Model / Source | Description |
|---|:---:|---|---|
| `clap_embedding` | 1024 (corrected 2026-09-15; was documented as 512, never verified against a real embedding) | `msclap` (LAION CLAP 2023) | Global acoustic context: instrumentation, energy, mood, production style. Strongest single signal. |
| `nlp_lyrics_vector` | 768 (corrected 2026-09-15; was documented as 384) | `paraphrase-multilingual-mpnet-base-v2` (SentenceTransformers) | Semantic meaning of Hindi/Urdu lyrics. Prevents acoustic matches that clash in meaning (e.g., a fast party song matching a grief song). |

**Lyrics handling rules:**

* Fetch lyrics using `lyricsgenius` (Genius API, free tier).
* Handle both Devanagari and Romanised Hindi input. The multilingual model accepts both.
* For Urdu-script lyrics (ghazals), preprocess with `urduhack` or pass raw — the multilingual model handles Urdu script adequately.
* If Genius returns no result: store a **zero vector** (384d of zeros), set `lyrics_missing=1` in SQLite, and log the song. Do not hallucinate or approximate lyrics.
* Older ghazals (Mehdi Hassan, Farida Khanum, Ghulam Ali) will frequently have no lyrics available. This is expected; the zero vector means the NLP component contributes nothing to their similarity, which is acceptable — their acoustic + Indian features are sufficient for correct clustering.

---

### Group 2 — Vocal / Gayaki Features

*Extracted from `clip_vocals.wav` (isolated vocal stem)*

| Feature | Dims | Model / Source | Description |
|---|:---:|---|---|
| `vocal_mfcc` | 20 | `librosa.feature.mfcc` on vocal stem | Captures the singer's physical vocal tract shape and timbre. Running on the isolated vocal stem (not full mix) produces cleaner coefficients, especially for ghazals where instrumentation masks vocals in the full mix. |
| `vocal_energy_ratio` | 1 | `librosa.feature.rms` (vocals) / `librosa.feature.rms` (instr) | **Most powerful single Indian-specific feature.** Ghazals score 0.8–0.9 (vocal-dominant). Party/club songs score 0.3–0.5 (beat-dominant). Sufi qawwalis score 0.5–0.7 (mixed). Directly fixes the Bhar Do Jholi Meri / Saturday Saturday confusion seen in UMAP. |
| `murki_index` | 1 | F0 modulation variance via `librosa.pyin` on vocal stem | Quantifies rapid classical ornamentation (murki, gamak). Computed as the variance of the F0 derivative, **restricted to oscillations between 5–15 Hz** to exclude vibrato (<5 Hz) and noise (>15 Hz). Higher values indicate classical ornamentation. Lower values indicate pop/filmi flat delivery. |

**murki\_index constraint:** The 5–15 Hz band filter is mandatory. Without it, vibrato-heavy singers (Lata Mangeshkar, Asha Bhosle) produce false high murki values despite not singing classical ornaments. Filter the F0 derivative in the frequency domain before computing variance.

---

### Group 3 — Melodic / Raga Features

*Extracted from full 90s clip (full mix, not separated)*

| Feature | Dims | Model / Source | Description |
|---|:---:|---|---|
| `microtonal_pcp` | 36 | `librosa.pyin` + custom binning OR `CREPE` | High-resolution pitch class profile using **36 bins per octave** instead of standard 12. Captures the 22 shrutis of Hindustani music that standard chroma collapses into 12 equal bins. Directly replaces `chroma_mean` in the feature vector. |
| `meend_variance` | 1 | First derivative of F0 from `librosa.pyin` | Captures the prevalence of pitch glides (meend, andolan). Classical singers glide continuously between pitches; pop/filmi singers hit discrete quantised steps. High meend\_variance = classical or semi-classical. Low = pop or electronic. |
| `raga_probability` | 30 | Essentia pretrained models / CompMusic Hindustani classifier | Probability distribution over **30 Hindustani ragas** (the 30 most common in Bollywood and classical repertoire — see list below). Groups songs by Rasa (emotional colour of the raga). Dimensionality is fixed at **30**. |

**raga\_probability implementation:**

Use the following in priority order:

1. **Essentia Music Extractor** (`essentia.standard.TonalExtractor`) — free, open source (MTG Barcelona). Has tonal descriptors that approximate raga-like pitch distributions. Not a direct raga classifier but usable as a proxy.
2. **CompMusic Hindustani Raga Recogniser** — open-source from the CompMusic project (UPF Barcelona / IIT Bombay). Trained on the Saraga dataset. Classifies 30 Hindustani ragas. Free to use for research.
3. **Fallback:** If neither model is available in the environment, substitute `raga_probability` with a **zero vector (30d)** and flag `raga_missing=1`. Do not skip the dimension — the vector schema must remain fixed at 30d.

**The 30 ragas covered:** Yaman, Bhairav, Bhairavi, Kafi, Khamaj, Bilawal, Kalyan, Marwa, Poorvi, Todi, Bhupali, Desh, Pilu, Tilak Kamod, Jhinjhoti, Pahadi, Kirwani, Charukeshi, Madhuvanti, Malkauns, Darbari, Bageshri, Kedar, Vrindavani Sarang, Miyan ki Malhar, Shuddh Sarang, Bhimpalasi, Jaunpuri, Lalit, Shree.

**microtonal\_pcp compute note:** `CREPE` is more accurate but slow on CPU (~45s per 90s clip). `librosa.pyin` is faster (~8s) with slightly lower pitch accuracy. Use `librosa.pyin` for batch processing on laptop and Kaggle CPU. Use `CREPE` for validation runs only.

---

### Group 4 — Rhythmic / Taal Features

*Extracted from `clip_instr.wav` (instrumental stem)*

| Feature | Dims | Model / Source | Description |
|---|:---:|---|---|
| `onset_skewness` | 1 | `scipy.stats.skew` on `librosa.onset.onset_strength` envelope | **Differentiates tabla/dholak from synthetic 808 beats.** Acoustic percussions (tabla, dholak) produce onset envelopes with positive skew due to the decay tail of skin drums. Synthetic 808 beats produce near-zero skew (sharp attack, no tail). This is the primary signal for separating Badshah from Jagjit Singh at the rhythmic level. |
| `tempo_bucket` | 4 | `librosa.beat.beat_track` → one-hot encoding | Hard-segments songs into 4 tempo zones: **Slow** (<80 BPM — ghazal/classical), **Mid** (80–100 BPM — romantic ballad), **Up-tempo** (100–120 BPM — indie/semi-fast), **Fast** (>120 BPM — party/dance). One-hot encoded. |
| `hnr_mean` | 1 | `librosa.effects.harmonic` on instrumental stem | Harmonic-to-Noise ratio of the instrumental stem. High HNR = melodic/acoustic instrumentation (sarangi, tabla, harmonium). Low HNR = noisy/percussive instrumentation (heavy 808, distorted synths). Separates acoustic classical from electronic club. |

---

## ⚖️ Fusion & Weighting

All feature groups must be independently L2-normalised **before** concatenation. This prevents high-dimensional groups from numerically dominating lower-dimensional ones. After normalisation, apply explicit scalar weights before concatenation, then L2-normalise the final fused vector.

### Weights (v1 — Pilot Defaults)

| Group | Features | Weight |
|---|---|:---:|
| CLAP neural | `clap_embedding` | 0.40 |
| NLP lyric | `nlp_lyrics_vector` | 0.15 |
| Vocal / Gayaki | `vocal_mfcc`, `vocal_energy_ratio`, `murki_index` | 0.20 |
| Melodic / Raga | `microtonal_pcp`, `meend_variance`, `raga_probability` | 0.15 |
| Rhythmic / Taal | `onset_skewness`, `tempo_bucket`, `hnr_mean` | 0.10 |

**Important:** When `lyrics_missing=1`, redistribute the NLP weight (0.15) proportionally to the other groups — do not leave it as zeros pulling the cosine similarity down. Recommended redistribution: add 0.08 to CLAP weight, 0.04 to Vocal, 0.03 to Melodic.

**Important:** When `raga_missing=1`, treat the raga probability group as if it doesn't exist for that song. Zero-padding with 30d zeros is fine for the vector schema, but do not include its weight in redistribution (it's already part of the Melodic group weight).

These weights are starting defaults. Re-evaluate after each UMAP visualisation pass. If ghazals and sufi songs are still merging, increase Vocal weight. If party and upbeat songs are merging, increase Rhythmic weight.

---

## 📐 Dimensionality Reduction

### The Problem

**Correction (2026-09-15):** this table's `clap_embedding` (512) and
`nlp_lyrics_vector` (384) dims were never checked against a real model
output and are both wrong — msclap's CLAP(version="2023") is actually
1024d, and `paraphrase-multilingual-mpnet-base-v2` is actually 768d (see
`pipeline/embedder.py`'s `CLAP_DIM` constant and module docstring for the
current, verified numbers, including `tonnetz_mean` which this table
predates). The rest of this table is kept for historical context on the
original design intent, not as an accurate current spec.

The raw fused vector was originally spec'd at ~991 dimensions, based on the
wrong dims above; actual raw dimension is ~1893 (1024 CLAP + 768 NLP + 22
vocal + 73 melodic incl. tonnetz + 6 rhythmic):

| Group | Dims |
|---|---|
| clap\_embedding | 512 (WRONG — actually 1024, see correction above) |
| nlp\_lyrics\_vector | 384 (WRONG — actually 768, see correction above) |
| vocal\_mfcc | 20 |
| vocal\_energy\_ratio | 1 |
| murki\_index | 1 |
| microtonal\_pcp | 36 |
| meend\_variance | 1 |
| raga\_probability | 30 |
| onset\_skewness | 1 |
| tempo\_bucket | 4 |
| hnr\_mean | 1 |
| **Total** | **~991** |

A ~991d FAISS IndexFlatIP on 50,000 songs would be slow and memory-heavy. Additionally, UMAP visualisation of 991d vectors produces noisy layouts.

### The Solution — PCA to 256d

After all feature extraction is complete and the fused vector is built, apply PCA to reduce to 256 dimensions before saving to disk and before FAISS indexing.

**Rules:**

1. The PCA model is fit **once** on the complete pilot dataset (50 songs minimum, 200 songs recommended for stable principal components).
2. The fitted PCA model is saved to `index/pca_model.pkl` using `joblib`.
3. All subsequent songs — including those added later by the async worker — use `pca_model.transform()`, never `pca_model.fit()` or `pca_model.fit_transform()`.
4. If the pilot dataset is reprocessed from scratch (e.g., after changing feature extraction), the PCA model must be refit and the FAISS index must be fully rebuilt.
5. The 256d compressed vector is what gets stored in `data/embeddings/song_id.npy`. The raw 991d vector is never persisted to disk.

**Why 256d:** Retains >95% of variance for typical music embedding datasets at this scale. Fast for both FAISS IndexFlatIP (pilot) and IndexIVFPQ (scale). UMAP runs well on 256d input.

---

## ⚙️ Extraction Flow

The full per-song extraction sequence in `processor.py`:

```
1.  Download audio (yt-dlp)
2.  Trim to 90s clip starting at clip_start (ffmpeg/pydub)
3.  DELETE raw download
4.  Run demucs (htdemucs_light) → clip_vocals.wav + clip_instr.wav
5.  Extract Group 2 (Vocal) features from clip_vocals.wav
6.  Extract Group 3 (Melodic) features from full clip
7.  Extract Group 4 (Rhythmic) features from clip_instr.wav
8.  Run CLAP embedding on full clip (Group 1)
9.  Fetch lyrics via Genius API
10. If lyrics found → run SentenceTransformer → save nlp/song_id.npy
    If not found    → save zero vector, set lyrics_missing=1
11. DELETE clip + stems (clip_vocals.wav, clip_instr.wav)
12. L2-normalise each feature group independently
13. Apply group weights, concatenate → ~991d fused vector
14. L2-normalise fused vector
15. PCA transform → 256d compressed vector
16. Save compressed vector to data/embeddings/song_id.npy
17. Save raw feature dict to data/features/song_id.json
18. Insert all paths + metadata to SQLite
```

---

## 🛠 Free Models & Tools Reference

| Feature | Recommended Tool | Alternative | Notes |
|---|---|---|---|
| Source separation | `demucs` htdemucs\_light (Meta, free) | `spleeter` (Deezer, free) | demucs: better quality, slower. spleeter: faster (~40s/clip on CPU), slightly noisier |
| Pitch tracking | `librosa.pyin` (built-in) | `CREPE` (Google/NYU, free) | pyin for batch; CREPE for validation only |
| Raga classification | CompMusic Hindustani classifier (free academic) | Essentia TonalExtractor (free) | CompMusic preferred; Essentia as fallback |
| NLP lyric embedding | `paraphrase-multilingual-mpnet-base-v2` (HuggingFace, free) | `ai4bharat/indic-bert` (free) | multilingual-mpnet for mixed scripts; indic-bert for Hindi-only |
| Lyrics fetch | `lyricsgenius` / Genius API (free tier) | Manual CSV supplement | Rate limit: ~1 req/sec. Use sleep(1) between requests |
| LLM (conversational) | Groq API free tier — `llama-3.1-70b-versatile` | Ollama local — `llama3.2:3b` | Groq for deployment; Ollama for offline dev |
| PCA | `sklearn.decomposition.PCA` (built-in) | — | Save with `joblib.dump` |
| FAISS | `faiss-cpu` (Meta, free) | — | IndexFlatIP (pilot), IndexIVFPQ (>1k songs) |

---

## 📊 Evaluation Metric

After building the FAISS index, run `scripts/evaluate.py` to compute **Precision@5 by category**:

For each song in the database, retrieve its top-5 recommendations. Record the fraction of those 5 that share the same category label. Average across all songs per category.

**Target benchmarks for the pilot dataset:**

| Category | Target Precision@5 |
|---|---|
| ghazal | ≥ 0.80 |
| sufi | ≥ 0.75 |
| sad\_romantic | ≥ 0.70 |
| party | ≥ 0.85 |
| upbeat | ≥ 0.65 |

If any category falls below its target, inspect its UMAP position and increase the weight of the feature group most relevant to that category's distinguishing characteristic.