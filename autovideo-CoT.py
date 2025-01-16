import os
import logging
import streamlit as st
from psycopg2 import connect
from psycopg2.extras import RealDictCursor
from celery import Celery
from rake_nltk import Rake
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips
from gtts import gTTS
from datetime import datetime
from urllib.parse import urlparse
import requests

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Celery Setup
celery_app = Celery("video_tasks", broker="redis://localhost:6379/0")

# PostgreSQL Database Connection
DB_CONN = connect(
    dbname="your_db_name",
    user="your_user",
    password="your_password",
    host="localhost",
    port="5432"
)
DB_CURSOR = DB_CONN.cursor(cursor_factory=RealDictCursor)

# Ensure necessary directories exist
os.makedirs("video_clips", exist_ok=True)
os.makedirs("output_videos", exist_ok=True)

# Database Table Initialization
DB_CURSOR.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id SERIAL PRIMARY KEY,
        user_id TEXT,
        text_input TEXT,
        logo_url TEXT,
        brand_name TEXT,
        style TEXT,
        output_url TEXT,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
DB_CONN.commit()

# Helper Functions
def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def download_video(url, filepath):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        return False

def retrieve_and_edit_videos(keywords, branding, style):
    video_clips = []
    for keyword in keywords:
        video_url = f"https://storage.googleapis.com/openimages/keywords/{keyword}.mp4"
        parsed_url = urlparse(video_url)
        video_filename = os.path.basename(parsed_url.path)
        video_filepath = os.path.join("video_clips", video_filename)

        if not os.path.exists(video_filepath):
            if not download_video(video_url, video_filepath):
                continue

        try:
            video_clip = VideoFileClip(video_filepath)
            video_clips.append(video_clip)
        except Exception as e:
            logging.error(f"Error processing video '{video_filename}': {e}")
            continue

    if not video_clips:
        logging.error("No valid videos found for provided keywords.")
        return None

    # Concatenate video clips
    final_video = concatenate_videoclips(video_clips, method="compose")

    # Add branding
    if branding.get("logo"):
        logo_clip = ImageClip(branding["logo"]).resize(height=50).set_duration(final_video.duration)
        final_video = CompositeVideoClip([final_video, logo_clip.set_position(("center", "top"))])

    return final_video

def generate_voiceover(text, language="en"):
    try:
        voiceover = gTTS(text=text, lang=language)
        voiceover_filename = f"voiceover_{datetime.now().timestamp()}.mp3"
        voiceover.save(voiceover_filename)
        return voiceover_filename
    except Exception as e:
        logging.error(f"Error generating voiceover: {e}")
        return None

@celery_app.task
def generate_video_task(user_id, video_id, text_input, branding, style):
    try:
        keywords = extract_keywords(text_input)
        final_video = retrieve_and_edit_videos(keywords, branding, style)

        if not final_video:
            DB_CURSOR.execute(
                "UPDATE videos SET status = %s WHERE id = %s",
                ("failed", video_id)
            )
            DB_CONN.commit()
            return

        # Save final video
        output_path = os.path.join("output_videos", f"output_{video_id}.mp4")
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Update database
        DB_CURSOR.execute(
            "UPDATE videos SET output_url = %s, status = %s WHERE id = %s",
            (output_path, "completed", video_id)
        )
        DB_CONN.commit()
    except Exception as e:
        logging.error(f"Error generating video task: {e}")
        DB_CURSOR.execute(
            "UPDATE videos SET status = %s WHERE id = %s",
            ("failed", video_id)
        )
        DB_CONN.commit()

# Streamlit App
st.title("Autovideo AI - Production Ready")

# User Inputs
text_input = st.text_area("Describe your video:", height=150)
uploaded_logo = st.file_uploader("Upload your logo (optional):", type=["png", "jpg", "jpeg"])
brand_name = st.text_input("Brand Name:")
style = st.selectbox("Video Style:", ["casual", "professional", "funny"])
submit = st.button("Generate Video")

if submit:
    if text_input.strip():
        branding = {
            "logo": None,
            "name": brand_name
        }

        if uploaded_logo:
            logo_path = os.path.join("video_clips", uploaded_logo.name)
            with open(logo_path, "wb") as f:
                f.write(uploaded_logo.read())
            branding["logo"] = logo_path

        DB_CURSOR.execute(
            "INSERT INTO videos (user_id, text_input, logo_url, brand_name, style) VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (None, text_input, uploaded_logo.name if uploaded_logo else None, brand_name, style)
        )
        video_id = DB_CURSOR.fetchone()["id"]
        DB_CONN.commit()

        generate_video_task.apply_async(args=(None, video_id, text_input, branding, style))
        st.success("Video generation has started. Check back soon!")

# Display Generated Videos
st.subheader("Generated Videos")
DB_CURSOR.execute("SELECT id, output_url, status FROM videos ORDER BY created_at DESC")
videos = DB_CURSOR.fetchall()

for video in videos:
    video_id = video["id"]
    output_url = video["output_url"]
    status = video["status"]

    if status == "completed":
        st.video(output_url)
    elif status == "pending":
        st.text(f"Video {video_id} is being generated...")
    elif status == "failed":
        st.error(f"Video {video_id} generation failed.")
