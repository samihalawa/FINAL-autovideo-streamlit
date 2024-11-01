import streamlit as st
from openai import OpenAI
import os
import moviepy.editor as mpe
import requests
from tempfile import NamedTemporaryFile
import logging
from gtts import gTTS
from dotenv import load_dotenv
import tempfile
import json
import time
import shutil
import numpy as np
import random
from pathlib import Path
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config at the very beginning of the script
st.set_page_config(page_title="AutovideoAI", page_icon="🎥", layout="wide")

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sample prompts
SAMPLE_PROMPTS = [
    "Create a motivational video about overcoming challenges",
    "Make an educational video explaining photosynthesis",
    "Design a funny video about the struggles of working from home",
]

# Update MUSIC_TRACKS with more reliable, royalty-free sources
MUSIC_TRACKS = {
    "Electronic": "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Tours/Enthusiast/Tours_-_01_-_Enthusiast.mp3",
    "Experimental": "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chad_Crouch/Arps/Chad_Crouch_-_Shipping_Lanes.mp3",
    "Folk": "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Kai_Engel/Satin/Kai_Engel_-_07_-_Interlude.mp3",
    "Hip-Hop": "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Kai_Engel/Sustains/Kai_Engel_-_08_-_Sentinel.mp3",
    "Instrumental": "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Kai_Engel/Sustains/Kai_Engel_-_03_-_Contention.mp3",
}

# Add conditional imports for video features
ENABLE_VIDEO = False
try:
    import moviepy.editor as mpe
    from gtts import gTTS
    from pydub import AudioSegment
    ENABLE_VIDEO = True
except ImportError:
    pass

def select_background_music(genre):
    try:
        if genre in MUSIC_TRACKS:
            track_url = MUSIC_TRACKS[genre]
        else:
            track_url = random.choice(list(MUSIC_TRACKS.values()))
        
        response = requests.get(track_url)
        if response.status_code == 200:
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_audio_file.write(response.content)
            temp_audio_file.close()
            return temp_audio_file.name
    except Exception as e:
        logger.error(f"Error selecting background music: {str(e)}")
    
    return None

def generate_storyboard(prompt, style="motivational"):
    try:
        input_text = f"""Generate a detailed {style} video storyboard based on this prompt: "{prompt}"
        Provide the storyboard in JSON format with the following structure:
        {{
            "title": "Overall video title",
            "scenes": [
                {{
                    "scene_number": 1,
                    "title": "Scene title",
                    "description": "Detailed scene description",
                    "narration": "Narration text for the scene",
                    "keywords": ["keyword1", "keyword2", "keyword3"],
                    "duration": "Duration in seconds",
                    "visual_elements": ["List of visual elements to include"],
                    "transitions": {{
                        "in": "Transition type for entering the scene",
                        "out": "Transition type for exiting the scene"
                    }}
                }}
            ],
            "target_audience": "Description of the target audience",
            "overall_tone": "Description of the overall tone of the video"
        }}
        Ensure there are at least 3 scenes in the storyboard."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a creative video storyboard generator. Respond with valid JSON following the specified structure."},
                {"role": "user", "content": input_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        storyboard = json.loads(response.choices[0].message.content)
        return storyboard
    except Exception as e:
        logger.error(f"Error generating storyboard: {str(e)}")
        st.error("An error occurred while generating the storyboard. Please try again.")
    return None

def create_scene_clip(scene, duration=5):
    try:
        # Create a gradient background
        gradient = np.linspace(0, 255, 1280)
        background = np.tile(gradient, (720, 1)).astype(np.uint8)
        background = np.stack((background,) * 3, axis=-1)
        
        # Create video clip from the background
        clip = mpe.ImageClip(background).set_duration(duration)
        
        # Add text
        txt_clip = mpe.TextClip(scene['title'], fontsize=70, color='white', font='Arial-Bold')
        txt_clip = txt_clip.set_position('center').set_duration(duration)
        
        # Add keywords as subtitles
        if 'keywords' in scene:
            keywords_txt = ", ".join(scene['keywords'])
            keywords_clip = mpe.TextClip(keywords_txt, fontsize=30, color='yellow', font='Arial')
            keywords_clip = keywords_clip.set_position(('center', 0.8), relative=True).set_duration(duration)
            clip = mpe.CompositeVideoClip([clip, txt_clip, keywords_clip])
        else:
            clip = mpe.CompositeVideoClip([clip, txt_clip])
        
        # Add fade in and out
        clip = clip.fadein(0.5).fadeout(0.5)
        
        return clip
    except Exception as e:
        logger.error(f"Error creating scene clip: {str(e)}")
        return mpe.ColorClip(size=(1280, 720), color=(0,0,0)).set_duration(duration)

def generate_voiceover(narration_text):
    logger.info(f"Generating voiceover for text: {narration_text[:50]}...")
    try:
        tts = gTTS(text=narration_text, lang='en', slow=False)
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_audio_file.name)
        return mpe.AudioFileClip(temp_audio_file.name)
    except Exception as e:
        logger.error(f"Error generating voiceover: {str(e)}")
        return mpe.AudioClip(lambda t: 0, duration=len(narration_text.split()) / 2)  # Silent audio

@contextmanager
def cleanup_context():
    """Context manager for cleaning up temporary files."""
    temp_files = []
    try:
        yield temp_files
    finally:
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.unlink(file)
            except Exception as e:
                logger.error(f"Failed to cleanup {file}: {e}")

def create_video(storyboard, background_music_file):
    with cleanup_context() as temp_files:
        try:
            clips = []
            for scene in storyboard['scenes']:
                clip = create_scene_clip(scene, float(scene['duration']))
                narration = generate_voiceover(scene['narration'])
                clip = clip.set_audio(narration)
                clips.append(clip)
            
            final_clip = mpe.concatenate_videoclips(clips)
            
            if background_music_file:
                background_music = mpe.AudioFileClip(background_music_file).volumex(0.1)
                background_music = background_music.audio_loop(duration=final_clip.duration)
                final_audio = mpe.CompositeAudioClip([final_clip.audio, background_music])
                final_clip = final_clip.set_audio(final_audio)
            
            output_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.mp4'
            ).name
            temp_files.append(output_file)
            final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac', fps=24)
            return output_file
        except Exception as e:
            logger.error(f"Error in create_video: {e}")
            return None

def main():
    if not ENABLE_VIDEO:
        st.error("Video features not available. Install required packages.")
        return
    st.title("AutovideoAI")
    
    # Store generated content in session state
    if 'generated_videos' not in st.session_state:
        st.session_state.generated_videos = {}
        
    # Use memory buffer instead of temp files
    if 'current_video' not in st.session_state:
        st.session_state.current_video = None
        
    prompt = st.text_input("Enter your video prompt:", placeholder="Create a motivational video about overcoming challenges")
    
    if st.button("Generate Video"):
        with st.spinner("Generating storyboard..."):
            storyboard = generate_storyboard(prompt)
        
        if storyboard:
            st.success("Storyboard generated successfully!")
            st.json(storyboard)
            
            music_style = st.selectbox("Select background music style:", list(MUSIC_TRACKS.keys()))
            
            if st.button("Create Video"):
                with st.spinner("Creating video..."):
                    background_music = select_background_music(music_style)
                    video_file = create_video(storyboard, background_music)
                
                if video_file:
                    st.success("Video created successfully!")
                    st.video(video_file)
                    
                    with open(video_file, "rb") as file:
                        st.download_button(
                            label="Download Video",
                            data=file,
                            file_name="generated_video.mp4",
                            mime="video/mp4"
                        )
        else:
            st.error("Failed to generate storyboard. Please try again.")

if __name__ == "__main__":
    main()