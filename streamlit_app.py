import streamlit as st
from openai import OpenAI
import os, logging, tempfile, json, time, shutil, random, gc, sys
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

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Error: OPENAI_API_KEY environment variable not found. Please set it in your .env file.")
    st.stop()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "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")
if not HUGGINGFACE_API_KEY:
    st.error("Error: HUGGINGFACE_API_KEY environment variable not found.")
    st.stop()

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    st.stop()

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
        if response.status_code != 200 or 'audio' not in response.headers.get('content-type', ''):
            logger.error(f"Invalid music track response for {genre}: Status {response.status_code}, Content-Type: {response.headers.get('content-type')}")
            st.warning(f"Could not load music track for {genre}. Using default audio.")
            return None
            
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.write(response.content)
        temp_file.close()
        
        try:
            audio_clip = mpe.AudioFileClip(temp_file.name)
            if audio_clip.duration < 1:
                raise ValueError("Audio file too short")
            audio_clip.close()
            return temp_file.name
        except Exception as e:
            logger.error(f"Error validating audio file: {e}")
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            return None
            
    except Exception as e:
        logger.error(f"Error selecting music: {e}")
        st.warning("Could not load background music. Continuing without music.")
        return None

def validate_storyboard(storyboard):
    """Validate the structure and content of a storyboard"""
    try:
        if not isinstance(storyboard, dict):
            raise ValueError("Invalid storyboard format: not a dictionary")
            
        # Check required top-level fields
        required_fields = ["title", "scenes"]
        missing_fields = [f for f in required_fields if f not in storyboard]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
        # Validate title
        if not isinstance(storyboard["title"], str) or len(storyboard["title"].strip()) == 0:
            raise ValueError("Invalid or empty title")
            
        # Validate scenes
        if not isinstance(storyboard["scenes"], list) or len(storyboard["scenes"]) == 0:
            raise ValueError("Storyboard must contain at least one scene")
            
        total_duration = 0
        for i, scene in enumerate(storyboard["scenes"]):
            # Check scene structure
            if not isinstance(scene, dict):
                raise ValueError(f"Scene {i+1} is not a dictionary")
                
            # Check required scene fields
            required_scene_fields = ["title", "description", "narration", "duration"]
            missing_scene_fields = [f for f in required_scene_fields if f not in scene]
            if missing_scene_fields:
                raise ValueError(f"Scene {i+1} missing required fields: {', '.join(missing_scene_fields)}")
                
            # Validate text fields
            for field in ["title", "description", "narration"]:
                if not isinstance(scene[field], str) or len(scene[field].strip()) == 0:
                    raise ValueError(f"Scene {i+1} has invalid or empty {field}")
                    
            # Validate and normalize duration
            try:
                duration = float(scene["duration"])
                if duration <= 0:
                    logger.warning(f"Scene {i+1} has invalid duration, setting to default")
                    scene["duration"] = 5.0
                elif duration > 30:
                    logger.warning(f"Scene {i+1} duration too long, capping at 30 seconds")
                    scene["duration"] = 30.0
                else:
                    scene["duration"] = duration
            except (ValueError, TypeError):
                logger.warning(f"Scene {i+1} has invalid duration, setting to default")
                scene["duration"] = 5.0
                
            total_duration += scene["duration"]
            
        # Check total duration
        if total_duration > 180:  # 3 minutes max
            raise ValueError(f"Total video duration ({total_duration}s) exceeds maximum limit of 180s")
            
        return True, storyboard
        
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        logger.error(f"Unexpected error in storyboard validation: {e}")
        return False, "Unexpected error in storyboard validation"

def generate_storyboard(prompt, style="motivational"):
    try:
        from prompts.autovideo_prompt import SYSTEM_PROMPT
        
        # Validate inputs
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Empty prompt provided")
            
        if not style or style not in [s.lower() for s in VIDEO_STYLES.keys()]:
            style = "motivational"  # Default to motivational if invalid style
            
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Based on prompt: "{prompt}" and style: "{style}", generate video storyboard...'}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        try:
            storyboard = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in storyboard: {e}")
            st.error("Error parsing storyboard response. Please try again.")
            return None
            
        # Validate the storyboard
        is_valid, result = validate_storyboard(storyboard)
        if not is_valid:
            logger.error(f"Storyboard validation failed: {result}")
            st.error(f"Invalid storyboard generated: {result}")
            return None
            
        return result
        
    except Exception as e:
        logger.error(f"Storyboard generation error: {e}")
        st.error("An unexpected error occurred while generating the storyboard. Please try again.")
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
                if hasattr(clip, 'close'):
                    clip.close()
                if hasattr(clip, 'audio') and hasattr(clip.audio, 'close'):
                    clip.audio.close()
            except Exception as e:
                logger.error(f"Error closing clip: {e}")
        
        # Then delete temp files
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
                    logger.info(f"Cleaned up temp file: {f}")
            except Exception as e:
                logger.error(f"Cleanup error for {f}: {e}")
        
        # Clear memory
        gc.collect()

def cleanup_session_files():
    """Clean up any temporary files from the session"""
    if 'temp_files' in st.session_state:
        for f in st.session_state.temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
                    logger.info(f"Cleaned up session file: {f}")
            except Exception as e:
                logger.error(f"Session cleanup error for {f}: {e}")
        st.session_state.temp_files = set()

def register_cleanup_handler():
    """Register cleanup handler for the session"""
    if not st.session_state.get('cleanup_registered'):
        st.session_state.cleanup_registered = True
        cleanup_session_files()
        # Register cleanup for when the script reruns
        st.session_state.temp_files = set()
        
def cleanup_old_files(max_age_hours=24):
    """Clean up old temporary files"""
    try:
        temp_dir = tempfile.gettempdir()
        current_time = time.time()
        
        for filename in os.listdir(temp_dir):
            if filename.endswith('.mp4') or filename.endswith('.mp3'):
                filepath = os.path.join(temp_dir, filename)
                try:
                    # Check file age
                    file_age = current_time - os.path.getctime(filepath)
                    if file_age > (max_age_hours * 3600):
                        os.unlink(filepath)
                        logger.info(f"Cleaned up old file: {filepath}")
                except Exception as e:
                    logger.error(f"Error cleaning up old file {filepath}: {e}")
    except Exception as e:
        logger.error(f"Error during old files cleanup: {e}")

def create_video(storyboard, background_music_file):
    with enhanced_cleanup_context() as (temp_files, clips):
        try:
            if not storyboard or 'scenes' not in storyboard:
                raise ValueError("Invalid storyboard format")
                
            clips = []
            for scene_idx, scene in enumerate(storyboard['scenes']):
                try:
                    clip = create_enhanced_scene_clip(
                        scene, 
                        get_cached_style_config(st.session_state.video_style),
                        float(scene.get('duration', 5))
                    )
                    
                    if not clip:
                        raise ValueError(f"Failed to create clip for scene {scene_idx + 1}")
                    
                    narration_file, _ = generate_voiceover(scene['narration'])
                    if narration_file:
                        temp_files.append(narration_file)
                        narration = mpe.AudioFileClip(narration_file)
                        clip = clip.set_audio(narration)
                        clips.append(clip)
                    else:
                        logger.warning(f"No narration for scene {scene_idx + 1}")
                        clips.append(clip)
                except Exception as e:
                    logger.error(f"Error creating scene {scene_idx + 1}: {e}")
                    st.warning(f"Error in scene {scene_idx + 1}. Skipping...")
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
                    st.warning("Could not add background music. Continuing with narration only.")
            
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
            
            # Add to session state for cleanup
            st.session_state.temp_files.add(output_file)
            
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
    if not os.getenv("OPENAI_API_KEY"): 
        issues.append("OpenAI API key not found")
    if not os.getenv("HUGGINGFACE_API_KEY"):
        issues.append("Hugging Face API key not found")
    for pkg in ['moviepy', 'gtts']:
        try: __import__(pkg)
        except ImportError: issues.append(f"{pkg} not installed")
    return issues

def check_disk_space(required_mb=500):
    try:
        _, _, free = shutil.disk_usage(tempfile.gettempdir())
        free_mb = free // (2**20)
        if free_mb < required_mb:
            st.error(f"Insufficient disk space. Required: {required_mb}MB, Available: {free_mb}MB")
            st.stop()
        return True
    except Exception as e:
        st.error(f"Error checking disk space: {str(e)}")
        st.stop()
        return False

def initialize_session_state():
    # Check disk space first
    check_disk_space()
    
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
    register_cleanup_handler()
    cleanup_old_files()  # Clean up old files at startup
    
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
            options=["Creative Commons Music", "User Upload", "None"],
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