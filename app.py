import streamlit as st
import sys
import os
import json
import tempfile

sys.path.append(".")
from utils.db import get_all_songs
from index.search import find_similar_by_name

# --- HELPER FUNCTIONS ---
def decode_tempo_bucket(bucket: list) -> str:
    """Translates the one-hot tempo bucket into a UI-friendly Vibe."""
    if not bucket or len(bucket) < 4: return "Unknown"
    if bucket[0] == 1: return "Slow / Ghazal 🌙"
    if bucket[1] == 1: return "Mid / Romantic 💖"
    if bucket[2] == 1: return "Up-Tempo / Groove 🎸"
    if bucket[3] == 1: return "Fast / Party 🔥"
    return "Unknown"

@st.cache_data
def load_data():
    all_songs = get_all_songs()
    # UPGRADE 1: Strictly validate BOTH neural and acoustic feature files exist
    valid_songs = []
    for s in all_songs:
        emb_path = s[6]
        feat_path = s[7]
        if emb_path and os.path.exists(emb_path) and feat_path and os.path.exists(feat_path):
            valid_songs.append(s)
    return valid_songs

# --- PAGE SETUP ---
st.set_page_config(page_title="Hindi Music Engine", layout="wide", page_icon="🎧")
st.title("🎧 Hindi Music Similarity Engine")
st.markdown("Discover visually and mathematically similar Hindi tracks using Demucs Stemming, NLP, and Indian Acoustic AI.")

songs = load_data()
song_map = {f"{s[1]} - {s[2]}": s for s in songs}

# --- TABS FOR DIFFERENT SEARCH MODES ---
tab1, tab2 = st.tabs(["📚 Search from Database", "🎙️ Upload New Audio"])

# ==========================================
# TAB 1: DATABASE SEARCH
# ==========================================
with tab1:
    st.markdown("### Select a Seed Song")
    selected_song = st.selectbox("Choose a track to base your recommendations on:", list(song_map.keys()))

    if st.button("Find Similar Tracks", type="primary"):
        title, artist = selected_song.split(" - ", 1)
        
        with st.spinner('Searching the AI brain...'):
            try:
                results = find_similar_by_name(title, artist, top_k=5)
            except Exception as e:
                st.error(f"Error during search: {e}")
                results = []
            
        if results:
            st.write("---")
            st.subheader(f"Because you listen to {title} by {artist}:")
            orig_url = song_map[selected_song][3]
            if orig_url:
                st.video(orig_url)
                
            st.write("---")
            st.subheader("🔥 Top AI Matches")
            
            # --- UPGRADE 2: DISPLAY ACOUSTIC DNA ---
            cols = st.columns(len(results))
            for i, (col, r) in enumerate(zip(cols, results)):
                with col:
                    st.write(f"### {i+1}. {r['title']}")
                    st.caption(f"by {r['artist']}")
                    
                    # Fetch database row for this result
                    rec_song_data = next((s for s in songs if s[0] == r['song_id']), None)
                    
                    if rec_song_data:
                        # Load the custom Indian features JSON to display cool stats
                        feat_path = rec_song_data[7]
                        try:
                            with open(feat_path, "r") as f:
                                features = json.load(f)
                                vibe = decode_tempo_bucket(features.get("tempo_bucket", []))
                                murki = features.get("murki_index", 0.0)
                                hnr = features.get("hnr_mean", 0.0)
                                
                                st.markdown(f"**Vibe:** {vibe}")
                                st.caption(f"🎤 Murki Index: {murki:.2f} | 🎹 HNR: {hnr:.2f}")
                        except Exception:
                            pass
                            
                        # Show Match Score
                        st.metric(label="Match Score", value=f"{r['score']:.4f}")
                        
                        # Show Video
                        rec_url = rec_song_data[3]
                        if rec_url:
                            st.video(rec_url)
                        else:
                            st.write("*(No YouTube URL available)*")

# ==========================================
# TAB 2: LIVE AUDIO UPLOAD (DEMUCS ON THE FLY)
# ==========================================
with tab2:
    st.markdown("### Discover matches for a song not in the database")
    st.info("⚠️ Uploading a new song will trigger Demucs Source Separation. This will take ~30-60 seconds to process on your Mac.")
    
    uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=['mp3', 'wav'])
    
    if uploaded_file is not None:
        if st.button("Process & Find Matches", type="primary", key="upload_btn"):
            with st.spinner('Separating vocals/instruments and extracting Indian acoustic features... Please wait!'):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Run the heavy pipeline on the uploaded file
                    # Live inference pipeline not yet implemented
                    # results = find_similar(query_audio_path=tmp_path, top_k=5)
                    results = []
                    
                    st.error("Live Demucs audio processing is not yet implemented in the backend! Stick to Tab 1 for now.")
                    st.write("---")
                    st.subheader("🔥 Top AI Matches for your Upload")
                    
                    cols = st.columns(len(results))
                    for i, (col, r) in enumerate(zip(cols, results)):
                        with col:
                            st.write(f"### {i+1}. {r['title']}")
                            st.caption(f"by {r['artist']}")
                            st.metric(label="Match Score", value=f"{r['score']:.4f}")
                            
                            rec_url = next((s[3] for s in songs if s[0] == r['song_id']), None)
                            if rec_url:
                                st.video(rec_url)
                                
                except Exception as e:
                    st.error(f"An error occurred during live processing: {e}")
                finally:
                    # Clean up the temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)