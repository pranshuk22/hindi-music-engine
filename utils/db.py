"""
utils/db.py
────────────
SQLite database helpers.

Schema (canonical per AGENTS.md Rule 3, extended per audit):
  id, title, artist, youtube_url, category, duration, tempo,
  embedding_path, features_path, nlp_path,
  lyrics_missing, raga_missing, stem_separation_failed,
  processed, created_at

WAL mode is enabled for parallel write safety.
All connections use a 15-second timeout to avoid "database is locked"
errors when ThreadPoolExecutor runs multiple pipeline workers.
"""

import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = "data/metadata.db"


# ── Connection factory ────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema setup ──────────────────────────────────────────────────────────────

def init_db():
    """
    Create the songs table if it does not exist, then run any pending
    column migrations for pre-existing databases.

    Safe to call multiple times — idempotent.
    """
    conn = _connect()
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id                     TEXT PRIMARY KEY,
            title                  TEXT NOT NULL,
            artist                 TEXT NOT NULL,
            youtube_url            TEXT,
            category               TEXT    DEFAULT 'unknown',
            duration               REAL,
            tempo                  REAL,
            embedding_path         TEXT,
            features_path          TEXT,
            nlp_path               TEXT,
            lyrics_missing         INTEGER DEFAULT 0,
            raga_missing           INTEGER DEFAULT 0,
            stem_separation_failed INTEGER DEFAULT 0,
            processed              INTEGER DEFAULT 0,
            created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Migrations: add columns that may be absent in pre-existing DBs ────────
    # Each entry is (column_name, sqlite_type_and_default).
    # ALTER TABLE ADD COLUMN is a no-op-safe operation in SQLite — we guard
    # it with an existence check to avoid noisy errors on fresh databases.
    existing_cols = {row[1] for row in c.execute("PRAGMA table_info(songs)")}

    migrations = [
        ("category",               "TEXT DEFAULT 'unknown'"),
        ("nlp_path",               "TEXT"),
        ("lyrics_missing",         "INTEGER DEFAULT 0"),
        ("raga_missing",           "INTEGER DEFAULT 0"),       # added: raga classifier unavailable
        ("stem_separation_failed", "INTEGER DEFAULT 0"),       # added: demucs fallback flag
    ]

    for col, typedef in migrations:
        if col not in existing_cols:
            logger.info("DB migration: adding column '%s'", col)
            c.execute(f"ALTER TABLE songs ADD COLUMN {col} {typedef}")

    conn.commit()
    conn.close()


# ── Write operations ──────────────────────────────────────────────────────────

def insert_song(
    song_id:               str,
    title:                 str,
    artist:                str,
    youtube_url:           str   = "",
    category:              str   = "unknown",
    duration:              float = None,
    tempo:                 float = None,
    embedding_path:        str   = None,
    features_path:         str   = None,
    nlp_path:              str   = None,
    lyrics_missing:        int   = 0,
    raga_missing:          int   = 0,
    stem_separation_failed: int  = 0,
):
    """
    Insert or replace a song record.

    INSERT OR REPLACE is idempotent — safe to call again after reprocessing
    (e.g. after fixing features and rebuilding embeddings from --from-features).

    Args:
        lyrics_missing:         1 when Genius returned no lyrics; NLP vector
                                is a zero vector and weights are redistributed.
        raga_missing:           1 when neither Essentia nor CompMusic classifier
                                was available; raga_probability is a zero vector.
        stem_separation_failed: 1 when demucs failed and fell back to the full
                                clip for both vocal and instrumental paths;
                                vocal_energy_ratio, murki_index, and
                                onset_skewness are degraded.
    """
    conn = _connect()
    conn.execute("""
        INSERT OR REPLACE INTO songs
          (id, title, artist, youtube_url, category, duration, tempo,
           embedding_path, features_path, nlp_path,
           lyrics_missing, raga_missing, stem_separation_failed,
           processed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    """, (
        song_id, title, artist, youtube_url, category, duration, tempo,
        embedding_path, features_path, nlp_path,
        int(lyrics_missing), int(raga_missing), int(stem_separation_failed),
    ))
    conn.commit()
    conn.close()


def mark_lyrics_missing(song_id: str) -> None:
    """Set lyrics_missing=1 for an already-inserted song."""
    conn = _connect()
    conn.execute("UPDATE songs SET lyrics_missing=1 WHERE id=?", (song_id,))
    conn.commit()
    conn.close()


def mark_raga_missing(song_id: str) -> None:
    """Set raga_missing=1 for an already-inserted song."""
    conn = _connect()
    conn.execute("UPDATE songs SET raga_missing=1 WHERE id=?", (song_id,))
    conn.commit()
    conn.close()


def mark_stem_failed(song_id: str) -> None:
    """Set stem_separation_failed=1 for an already-inserted song."""
    conn = _connect()
    conn.execute(
        "UPDATE songs SET stem_separation_failed=1 WHERE id=?", (song_id,)
    )
    conn.commit()
    conn.close()


# ── Read operations ───────────────────────────────────────────────────────────

def get_all_songs(processed_only: bool = True) -> list:
    """
    Return all songs as sqlite3.Row objects (access by column name).
    """
    conn = _connect()
    c    = conn.cursor()
    if processed_only:
        c.execute("SELECT * FROM songs WHERE processed=1 ORDER BY created_at")
    else:
        c.execute("SELECT * FROM songs ORDER BY created_at")
    rows = c.fetchall()
    conn.close()
    return rows


def get_song_by_id(song_id: str):
    """Return a single song row, or None if not found."""
    conn = _connect()
    c    = conn.cursor()
    c.execute("SELECT * FROM songs WHERE id=?", (song_id,))
    row  = c.fetchone()
    conn.close()
    return row


def get_songs_by_category(category: str) -> list:
    """Return all processed songs matching a category label."""
    conn = _connect()
    c    = conn.cursor()
    c.execute(
        "SELECT * FROM songs WHERE processed=1 AND category=? ORDER BY created_at",
        (category,)
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_all_categories() -> list:
    """Return sorted list of distinct category values for processed songs."""
    conn = _connect()
    c    = conn.cursor()
    c.execute("SELECT DISTINCT category FROM songs WHERE processed=1")
    cats = [row[0] for row in c.fetchall()]
    conn.close()
    return sorted(cats)


def song_exists(song_id: str) -> bool:
    """Return True if song_id is already processed in the DB."""
    conn = _connect()
    c    = conn.cursor()
    c.execute("SELECT 1 FROM songs WHERE id=? AND processed=1", (song_id,))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def get_flag_summary() -> dict:
    """
    Return counts of songs with each quality flag set.
    Useful for pipeline diagnostics and dashboard display.

    Returns:
        {
            "total":                  int,
            "lyrics_missing":         int,
            "raga_missing":           int,
            "stem_separation_failed": int,
        }
    """
    conn = _connect()
    c    = conn.cursor()
    c.execute("""
        SELECT
            COUNT(*)                              AS total,
            SUM(lyrics_missing)                   AS lyrics_missing,
            SUM(raga_missing)                     AS raga_missing,
            SUM(stem_separation_failed)           AS stem_separation_failed
        FROM songs WHERE processed=1
    """)
    row = c.fetchone()
    conn.close()
    return {
        "total":                  row["total"]                  or 0,
        "lyrics_missing":         row["lyrics_missing"]         or 0,
        "raga_missing":           row["raga_missing"]           or 0,
        "stem_separation_failed": row["stem_separation_failed"] or 0,
    }