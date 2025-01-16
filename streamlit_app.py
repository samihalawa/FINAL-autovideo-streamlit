import streamlit as st
from openai import OpenAI
import os, logging, tempfile, json, time, shutil, random, gc
import moviepy.editor as mpe
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from functools import lru_cache
import concurrent.futures
from gtts import gTTS
from dotenv import load_dotenv
import requests

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="AutovideoAI", page_icon="🎥", layout="wide")
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
MUSIC_TRACKS = {
    "Electronic": "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Tours/Enthusiast/Tours_-_01_-_Enthusiast.mp3",
    "Experimental": "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chad_Crouch/Arps/Chad_Crouch_-_Shipping_Lanes.mp3", 
    "Folk": "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Kai_Engel/Satin/Kai_Engel_-_07_-_Interlude.mp3",
    "Hip-Hop": "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Kai_Engel/Sustains/Kai_Engel_-_08_-_Sentinel.mp3",
    "Instrumental": "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Kai_Engel/Sustains/Kai_Engel_-_03_-_Contention.mp3"
}

VIDEO_STYLES = {
    "Motivational": {
        "description": "Inspiring and uplifting content with dynamic transitions",
        "default_duration": 15,
        "color_scheme": ["#FF6B6B", "#4ECDC4"],
        "transitions": ["fade", "slide", "zoom"]
    },
    "Educational": {
        "description": "Clear and structured content with clean transitions",
        "default_duration": 20,
        "color_scheme": ["#95E1D3", "#EAFFD0"],
        "transitions": ["fade", "slide"]
    },
    "Corporate": {
        "description": "Professional and polished look with subtle transitions",
        "default_duration": 12,
        "color_scheme": ["#2C3E50", "#ECF0F1"],
        "transitions": ["fade"]
    },
    "Creative": {
        "description": "Artistic and experimental with dynamic effects",
        "default_duration": 18,
        "color_scheme": ["#FF9A8B", "#FF6B6B"],
        "transitions": ["zoom", "slide", "fade", "rotate"]
    }
}

VOICE_OPTIONS = {
    "en-US-Standard": "Standard American English",
    "en-GB-Standard": "British English", 
    "en-AU-Standard": "Australian English"
}

# Helper functions
@lru_cache(maxsize=32)
def get_cached_style_config(style_name):
    return VIDEO_STYLES.get(style_name, VIDEO_STYLES["Motivational"])

@st.cache_data
def load_sample_prompts():
    return {
        "Motivational": [
            "Create an inspiring video about never giving up on your dreams",
            "Design a motivational video about overcoming life's challenges"
        ],
        "Educational": [
            "Explain how climate change affects our planet",
            "Create a video about the basics of quantum physics"
        ],
        "Corporate": [
            "Present our company's quarterly achievements",
            "Introduce our new product features"
        ],
        "Creative": [
            "Tell a story about a magical forest",
            "Create an artistic video about the four seasons"
        ]
    }

def select_background_music(genre):
    try:
        track_url = MUSIC_TRACKS.get(genre, random.choice(list(MUSIC_TRACKS.values())))
        response = requests.get(track_url, timeout=10)
        if response.status_code == 200 and 'audio' in response.headers.get('content-type', ''):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.write(response.content)
            temp_file.close()
            try:
                mpe.AudioFileClip(temp_file.name)
                return temp_file.name
            except:
                os.unlink(temp_file.name)
        return None
    except Exception as e:
        logger.error(f"Error selecting music: {e}")
        return None

def generate_storyboard(prompt, style="motivational"):
    try:
        from prompts.autovideo_prompt import SYSTEM_PROMPT
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Based on prompt: "{prompt}" and style: "{style}", generate video storyboard...'}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Storyboard generation error: {e}")
        return None

def create_enhanced_scene_clip(scene, style_config, duration=5):
    try:
        width, height = 1920, 1080
        color1, color2 = style_config["color_scheme"]
        gradient = np.tile(np.linspace(0, 1, width)[:, np.newaxis] * np.array([int(color2[i:i+2], 16) - int(color1[i:i+2], 16) for i in (1,3,5)]) + np.array([int(color1[i:i+2], 16) for i in (1,3,5)]), (height, 1, 1)).astype(np.uint8)
        
        clip = mpe.ImageClip(gradient).set_duration(duration)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            title_clip = executor.submit(lambda: mpe.TextClip(scene['title'], fontsize=70, color='white').set_position(('center', 0.3)).set_duration(duration))
            desc_clip = executor.submit(lambda: mpe.TextClip(scene['description'][:97] + "..." if len(scene['description']) > 100 else scene['description'], fontsize=30, color='white').set_position(('center', 0.6)).set_duration(duration))
            
            return mpe.CompositeVideoClip([clip, title_clip.result(), desc_clip.result()])
    except Exception as e:
        logger.error(f"Scene clip creation error: {e}")
        return mpe.ColorClip(size=(1920, 1080), color=(0,0,0)).set_duration(duration)

@contextmanager
def enhanced_cleanup_context():
    temp_files = []
    clips = []
    try:
        yield temp_files, clips
    finally:
        # Close all clips first
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        # Then delete temp files
        for f in temp_files:
            try:
                if os.path.exists(f): os.unlink(f)
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
        gc.collect()

def create_video(storyboard, background_music_file):
    with enhanced_cleanup_context() as (temp_files, clips):
        try:
            if not storyboard or 'scenes' not in storyboard:
                raise ValueError("Invalid storyboard format")
                
            clips = []
            for scene in storyboard['scenes']:
                try:
                    clip = create_enhanced_scene_clip(
                        scene, 
                        get_cached_style_config(st.session_state.video_style),
                        float(scene.get('duration', 5))
                    )
                    
                    if clip:
                        narration_file, _ = generate_voiceover(scene['narration'])
                        if narration_file:
                            temp_files.append(narration_file)
                            narration = mpe.AudioFileClip(narration_file)
                            clip = clip.set_audio(narration)
                            clips.append(clip)
                except Exception as e:
                    logger.error(f"Scene creation error: {e}")
                    continue
            
            if not clips:
                raise Exception("No valid video clips could be generated")
                
            final_clip = mpe.concatenate_videoclips(clips)
            
            if background_music_file:
                try:
                    bg_music = mpe.AudioFileClip(background_music_file).volumex(0.1)
                    bg_music = bg_music.loop(duration=final_clip.duration)
                    final_clip = final_clip.set_audio(
                        mpe.CompositeAudioClip([final_clip.audio, bg_music])
                    )
                    clips.append(bg_music)
                except Exception as e:
                    logger.warning(f"Background music error: {e}")
            
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            temp_files.append(output_file)
            
            final_clip.write_videofile(
                output_file,
                codec='libx264',
                audio_codec='aac',
                fps=24,
                logger=None
            )
            clips.append(final_clip)
            return output_file
            
        except Exception as e:
            logger.error(f"Video creation error: {e}")
            st.error(f"Error creating video: {str(e)}")
            return None

def generate_voiceover(text):
    try:
        tts = gTTS(text=text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        return temp_file.name, temp_file.name
    except Exception as e:
        logger.error(f"Voiceover error: {e}")
        st.error("Error generating voiceover. Please try again.")
        return None, None

def verify_environment():
    issues = []
    if not os.getenv("OPENAI_API_KEY"): issues.append("OpenAI API key not found")
    for pkg in ['moviepy', 'gtts']:
        try: __import__(pkg)
        except ImportError: issues.append(f"{pkg} not installed")
    return issues

def check_disk_space(required_mb=500):
    try:
        _, _, free = shutil.disk_usage(tempfile.gettempdir())
        return free // (2**20) >= required_mb
    except Exception as e:
        logger.error(f"Disk space check error: {e}")
        return False

def initialize_session_state():
    defaults = {
        'initialized': True,
        'current_step': 1,
        'storyboard': None,
        'video_style': 'Motivational',
        'voice_option': list(VOICE_OPTIONS.keys())[0],
        'music_style': 'Electronic',
        'temp_files': set(),
        'last_prompt': None,
        'processing_error': None,
        'generation_complete': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Add validation decorator
def validate_environment(func):
    def wrapper(*args, **kwargs):
        issues = verify_environment()
        if issues:
            st.error("Setup Issues Detected:")
            for issue in issues:
                st.warning(issue)
            return
        if not check_disk_space():
            st.error("Insufficient disk space. Please free up at least 500MB.")
            return
        return func(*args, **kwargs)
    return wrapper

# Improved main UI layout
@validate_environment
def main():
    initialize_session_state()
    
    st.title("🎥 AutovideoAI")
    st.subheader("Create Professional Videos with AI")
    
    # Sidebar configuration
    with st.sidebar:
        progress_bar = st.progress(0)
        st.markdown("---")
        
        video_style = st.selectbox(
            "Select Video Style",
            options=list(VIDEO_STYLES.keys()),
            key="video_style"
        )
        
        voice_option = st.selectbox(
            "Select Voice",
            options=list(VOICE_OPTIONS.keys()),
            format_func=lambda x: VOICE_OPTIONS[x],
            key="voice_option"
        )
        
        music_style = st.selectbox(
            "Select Music Style",
            options=list(MUSIC_TRACKS.keys()),
            key="music_style"
        )

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_main_content(progress_bar)
        
    with col2:
        render_style_preview(video_style)

def render_main_content(progress_bar):
    if st.session_state.current_step == 1:
        render_step_one(progress_bar)
    elif st.session_state.current_step == 2:
        render_step_two(progress_bar)
    elif st.session_state.current_step == 3:
        render_step_three(progress_bar)

def render_step_one(progress_bar):
    st.subheader("1. Enter Your Video Prompt")
    sample_prompts = load_sample_prompts()
    selected_prompt = st.selectbox(
        "Or choose a sample prompt:",
        options=[""] + sample_prompts[st.session_state.video_style],
        key="sample_prompt"
    )
    
    prompt = st.text_area("Enter your video prompt:", value=selected_prompt, height=100)
    
    if st.button("Generate Storyboard", type="primary") and prompt:
        with st.spinner("Generating storyboard..."):
            st.session_state.storyboard = generate_storyboard(
                prompt,
                st.session_state.video_style.lower()
            )
            if st.session_state.storyboard:
                st.session_state.current_step = 2
                progress_bar.progress(0.33)

def render_step_two(progress_bar):
    st.subheader("2. Review and Customize")
    if st.session_state.storyboard:
        # Display storyboard in a more readable format
        with st.expander("View Storyboard Details", expanded=True):
            st.json(st.session_state.storyboard)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Generate Video", type="primary"):
                st.session_state.current_step = 3
                progress_bar.progress(0.66)
        with col2:
            if st.button("Back", type="secondary"):
                st.session_state.current_step = 1
                progress_bar.progress(0)
        with col3:
            if st.button("Regenerate Storyboard"):
                if st.session_state.last_prompt:
                    with st.spinner("Regenerating storyboard..."):
                        st.session_state.storyboard = generate_storyboard(
                            st.session_state.last_prompt,
                            st.session_state.video_style.lower()
                        )

def render_step_three(progress_bar):
    st.subheader("3. Generate Video")
    
    if not st.session_state.generation_complete:
        with st.spinner("Creating video..."):
            video_file = create_video(
                st.session_state.storyboard, 
                select_background_music(st.session_state.music_style)
            )
            if video_file:
                st.session_state.generation_complete = True
                st.session_state.video_file = video_file
                st.success("Video created successfully!")
                progress_bar.progress(1.0)
            else:
                st.error("Video generation failed")
                if st.button("Retry"):
                    st.session_state.current_step = 2
                    progress_bar.progress(0.33)
                return
    
    if st.session_state.generation_complete:
        st.video(st.session_state.video_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            with open(st.session_state.video_file, "rb") as f:
                st.download_button(
                    "Download Video",
                    f,
                    "video.mp4",
                    "video/mp4",
                    use_container_width=True
                )
        with col2:
            if st.button("Create Another Video", use_container_width=True):
                st.session_state.current_step = 1
                st.session_state.generation_complete = False
                st.session_state.storyboard = None
                progress_bar.progress(0)

def render_style_preview(video_style):
    st.subheader("Style Preview")
    
    # Display style information
    st.markdown(f"**{video_style}**")
    st.markdown(VIDEO_STYLES[video_style]['description'])
    
    # Display style features
    st.markdown("---")
    st.markdown("### Features")
    
    # Create two columns for features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("🎨 **Visual Style**")
        colors = VIDEO_STYLES[video_style]['color_scheme']
        st.markdown(f"- Primary: `{colors[0]}`")
        st.markdown(f"- Secondary: `{colors[1]}`")
        
        st.markdown("🎬 **Transitions**")
        for transition in VIDEO_STYLES[video_style]['transitions']:
            st.markdown(f"- {transition.title()}")
    
    with col2:
        st.markdown("⏱️ **Duration**")
        st.markdown(f"- Default: {VIDEO_STYLES[video_style]['default_duration']}s")
        
        st.markdown("🎵 **Audio**")
        st.markdown("- AI narration")
        st.markdown("- Background music")

if __name__ == "__main__":
    main()