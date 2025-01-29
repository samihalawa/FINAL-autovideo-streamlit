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
import re
import psutil

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

def validate_audio_file(file_path):
    try:
        audio = mpe.AudioFileClip(file_path)
        duration = audio.duration
        audio.close()
        return True if 0 < duration < 600 else False  # Max 10 minutes
    except Exception:
        return False

def select_background_music(genre):
    try:
        track_url = MUSIC_TRACKS.get(genre, random.choice(list(MUSIC_TRACKS.values())))
        response = requests.get(track_url, timeout=10)
        if response.status_code == 200 and 'audio' in response.headers.get('content-type', ''):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.write(response.content)
            temp_file.close()
            if validate_audio_file(temp_file.name):
                return temp_file.name
            os.unlink(temp_file.name)
        return None
    except Exception as e:
        logger.error(f"Error selecting music: {e}")
        return None

def validate_storyboard(storyboard):
    try:
        required_fields = ['scenes']
        if not all(field in storyboard for field in required_fields):
            return False
        
        for scene in storyboard['scenes']:
            if not all(field in scene for field in ['title', 'description', 'narration']):
                return False
        return True
    except Exception:
        return False

def sanitize_text(text):
    # Remove potentially problematic characters
    text = re.sub(r'[<>&;]', '', text)
    # Limit length
    return text[:500] if text else ""

def generate_storyboard(prompt, style="motivational"):
    try:
        prompt = sanitize_text(prompt)
        style = style.lower()
        if style not in [s.lower() for s in VIDEO_STYLES.keys()]:
            style = "motivational"
        
        if 'last_api_call' in st.session_state and time.time() - st.session_state.last_api_call < 1:
            time.sleep(1)
        st.session_state.last_api_call = time.time()
        
        if not prompt or len(prompt.strip()) < 10:
            raise ValueError("Prompt too short")
            
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
        storyboard = json.loads(response.choices[0].message.content)
        if not validate_storyboard(storyboard):
            raise ValueError("Invalid storyboard format")
        return storyboard
    except Exception as e:
        logger.error(f"Storyboard generation error: {e}")
        st.session_state.last_error = str(e)
        return None

def create_enhanced_scene_clip(scene, style_config, duration=5):
    try:
        width, height = 1920, 1080
        color1, color2 = style_config["color_scheme"]
        gradient = np.tile(np.linspace(0, 1, width)[:, np.newaxis] * np.array([int(color2[i:i+2], 16) - int(color1[i:i+2], 16) for i in (1,3,5)]) + np.array([int(color1[i:i+2], 16) for i in (1,3,5)]), (height, 1, 1)).astype(np.uint8)
        
        clip = mpe.ImageClip(gradient).set_duration(duration)
        
        title_clip = mpe.TextClip(scene['title'], fontsize=70, color='white').set_position(('center', 0.3)).set_duration(duration)
        desc_clip = mpe.TextClip(scene['description'][:97] + "..." if len(scene['description']) > 100 else scene['description'], fontsize=30, color='white').set_position(('center', 0.6)).set_duration(duration)
            
        return mpe.CompositeVideoClip([clip, title_clip, desc_clip])
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

def apply_transition(clip1, clip2, transition_type, duration=1.0):
    try:
        if transition_type == "fade":
            return mpe.CompositeVideoClip([
                clip1.crossfadein(duration),
                clip2.crossfadeout(duration)
            ])
        elif transition_type == "slide":
            return mpe.CompositeVideoClip([
                clip1,
                clip2.set_position(lambda t: (1920*(1-t/duration), 0))
            ])
        return clip1
    except Exception as e:
        logger.error(f"Transition error: {e}")
        return clip1

def save_generation_history(video_file):
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    history_entry = {
        'timestamp': time.time(),
        'style': st.session_state.video_style,
        'prompt': st.session_state.last_prompt,
        'video_path': video_file
    }
    st.session_state.generation_history.append(history_entry)
    st.session_state.last_successful_video = video_file

def check_memory_usage():
    memory = psutil.virtual_memory()
    return memory.percent < 90  # Return False if memory usage is above 90%

def create_video(storyboard, background_music_file):
    if not check_memory_usage():
        st.error("System memory is too low. Please try again later.")
        return None
        
    MAX_VIDEO_SIZE = 500 * 1024 * 1024
    with enhanced_cleanup_context() as (temp_files, clips):
        try:
            if background_music_file:
                temp_files.append(background_music_file)
            clips = []
            
            total_scenes = len(storyboard['scenes'])
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            for idx, scene in enumerate(storyboard['scenes']):
                progress_text.text(f"Processing scene {idx + 1}/{total_scenes}")
                progress_bar.progress((idx + 1) / total_scenes)
                
                clip = create_enhanced_scene_clip(scene, get_cached_style_config(st.session_state.video_style), float(scene.get('duration', 5)))
                if clip:
                    narration_file, _ = generate_voiceover(scene['narration'])
                    if narration_file:
                        temp_files.append(narration_file)
                        narration = mpe.AudioFileClip(narration_file)
                        clip = clip.set_audio(narration)
                        clips.append(clip)
                
            if not clips:
                raise Exception("No video clips could be generated")
                    
            # Apply transitions between clips
            style_config = get_cached_style_config(st.session_state.video_style)
            final_clips = []
            for i in range(len(clips)):
                if i > 0:
                    transition = random.choice(style_config["transitions"])
                    clips[i] = apply_transition(clips[i-1], clips[i], transition)
                final_clips.append(clips[i])
            
            final_clip = mpe.concatenate_videoclips(final_clips)
            if background_music_file:
                bg_music = mpe.AudioFileClip(background_music_file).volumex(0.1)
                bg_music = bg_music.loop(duration=final_clip.duration)
                final_clip = final_clip.set_audio(mpe.CompositeAudioClip([final_clip.audio, bg_music]))
                clips.append(bg_music)
            
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            temp_files.append(output_file)
            final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac', fps=24, logger=None)
            clips.append(final_clip)
            
            if os.path.getsize(output_file) > MAX_VIDEO_SIZE:
                raise ValueError("Generated video exceeds size limit")
                    
            save_generation_history(output_file)
            return output_file
        except Exception as e:
            logger.error(f"Video creation error: {e}")
            st.error("Failed to create video. Please try again.")
            return None

def generate_voiceover(text):
    try:
        # Get selected voice option from session state
        voice_lang = st.session_state.voice_option.split('-')[0]  # Extract language code
        tts = gTTS(text=text, lang=voice_lang)
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
    if 'initialized' not in st.session_state:
        default_state = {
            'initialized': True,
            'current_step': 1,
            'storyboard': None,
            'video_style': 'Motivational',
            'voice_option': list(VOICE_OPTIONS.keys())[0],
            'music_style': 'Electronic',
            'temp_files': set(),
            'last_prompt': None,
            'processing_error': None,
            'last_error': None,
            'generation_history': [],
            'last_successful_video': None
        }
        st.session_state.update(default_state)
    else:
        # Ensure all required keys exist
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

def reset_session():
    for key in ['storyboard', 'last_prompt', 'processing_error', 'last_error']:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.current_step = 1

def check_resource_limits():
    MAX_SCENES = 10
    MAX_DURATION_PER_SCENE = 30
    MAX_TOTAL_DURATION = 180  # 3 minutes
    
    if st.session_state.storyboard:
        scenes = st.session_state.storyboard.get('scenes', [])
        if len(scenes) > MAX_SCENES:
            st.error(f"Too many scenes (max {MAX_SCENES})")
            return False
            
        total_duration = sum(float(scene.get('duration', 5)) for scene in scenes)
        if total_duration > MAX_TOTAL_DURATION:
            st.error(f"Total duration exceeds limit ({MAX_TOTAL_DURATION}s)")
            return False
            
        for scene in scenes:
            if float(scene.get('duration', 5)) > MAX_DURATION_PER_SCENE:
                st.error(f"Scene duration exceeds limit ({MAX_DURATION_PER_SCENE}s)")
                return False
    return True

def cleanup_old_cache():
    MAX_CACHE_AGE = 3600  # 1 hour
    MAX_CACHE_SIZE = 1024 * 1024 * 1024  # 1GB
    
    cache_dir = tempfile.gettempdir()
    current_time = time.time()
    
    try:
        for file in os.listdir(cache_dir):
            if file.endswith('.mp4') or file.endswith('.mp3'):
                file_path = os.path.join(cache_dir, file)
                if current_time - os.path.getctime(file_path) > MAX_CACHE_AGE:
                    try:
                        os.unlink(file_path)
                    except OSError:
                        pass
    except Exception as e:
        logger.error(f"Cache cleanup error: {e}")

def handle_error_state():
    if 'last_error' in st.session_state and st.session_state.last_error:
        error_message = st.session_state.last_error
        st.error(f"Previous error: {error_message}")
        if st.button("Clear Error"):
            st.session_state.last_error = None
            st.session_state.processing_error = None
            st.experimental_rerun()

def main():
    initialize_session_state()
    cleanup_old_cache()
    handle_error_state()
    issues = verify_environment()
    if issues:
        st.error("Setup Issues Detected:")
        for issue in issues: st.warning(issue)
        return
        
    if not check_disk_space():
        st.error("Insufficient disk space. Please free up at least 500MB.")
        return

    st.title("🎥 AutovideoAI")
    st.subheader("Create Professional Videos with AI")
    
    with st.sidebar:
        progress_bar = st.progress(0)
        st.markdown("---")
        video_style = st.selectbox("Select Video Style", options=list(VIDEO_STYLES.keys()))
        st.session_state.video_style = video_style
        voice_option = st.selectbox("Select Voice", options=list(VOICE_OPTIONS.keys()), format_func=lambda x: VOICE_OPTIONS[x])
        st.session_state.voice_option = voice_option
        music_style = st.selectbox("Select Music", options=list(MUSIC_TRACKS.keys()))
        st.session_state.music_style = music_style

    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.current_step == 1:
            st.subheader("1. Enter Your Video Prompt")
            sample_prompts = load_sample_prompts()
            selected_prompt = st.selectbox("Or choose a sample prompt:", options=[""] + sample_prompts[video_style], key="sample_prompt")
            prompt = st.text_area("Enter your video prompt:", value=selected_prompt, height=100)
            
            if st.button("Generate Storyboard", type="primary") and prompt:
                with st.spinner("Generating storyboard..."):
                    st.session_state.storyboard = generate_storyboard(prompt, video_style.lower())
                    if st.session_state.storyboard:
                        st.session_state.current_step = 2
                        progress_bar.progress(0.33)
            
        elif st.session_state.current_step == 2:
            st.subheader("2. Review and Customize")
            if st.session_state.storyboard:
                st.json(st.session_state.storyboard)
                if st.button("Generate Video", type="primary"):
                    st.session_state.current_step = 3
                    progress_bar.progress(0.66)
                if st.button("Back", type="secondary"):
                    st.session_state.current_step = 1
                    progress_bar.progress(0)
                    
        elif st.session_state.current_step == 3:
            st.subheader("3. Generate Video")
            with st.spinner("Creating video..."):
                if not check_resource_limits():
                    st.session_state.current_step = 2
                    return
                video_file = create_video(st.session_state.storyboard, select_background_music(st.session_state.get('music_style')))
                if video_file:
                    st.success("Video created!")
                    st.video(video_file)
                    progress_bar.progress(1.0)
                    
                    with open(video_file, "rb") as f:
                        st.download_button("Download Video", f, "video.mp4", "video/mp4")
                    
                    if st.button("Create Another"): 
                        st.session_state.current_step = 1
                        progress_bar.progress(0)
                else:
                    st.error("Generation failed")
                    if st.button("Retry"):
                        st.session_state.current_step = 2
                        progress_bar.progress(0.33)

    with col2:
        st.subheader("Style Preview")
        st.markdown(f"**{video_style}**\n{VIDEO_STYLES[video_style]['description']}")
        st.markdown("---\n### Features\n- Professional transitions\n- Custom color scheme\n- AI-powered narration\n- Royalty-free music")

    if st.sidebar.button("Reset"):
        reset_session()

if __name__ == "__main__":
    main()