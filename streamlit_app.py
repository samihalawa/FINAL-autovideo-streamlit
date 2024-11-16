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
from functools import lru_cache
import gc
import concurrent.futures

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

# Add new constants
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

# Add voice options
VOICE_OPTIONS = {
    "en-US-Standard": "Standard American English",
    "en-GB-Standard": "British English",
    "en-AU-Standard": "Australian English"
}

@lru_cache(maxsize=32)
def get_cached_style_config(style_name):
    """Cache style configurations to reduce memory usage"""
    return VIDEO_STYLES.get(style_name, VIDEO_STYLES["Motivational"])

@st.cache_data
def load_sample_prompts():
    """Load and cache sample prompts for different video styles."""
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
        if genre in MUSIC_TRACKS:
            track_url = MUSIC_TRACKS[genre]
        else:
            track_url = random.choice(list(MUSIC_TRACKS.values()))
        
        # Add timeout and verify content type
        response = requests.get(track_url, timeout=10)
        if response.status_code == 200 and 'audio' in response.headers.get('content-type', ''):
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_audio_file.write(response.content)
            temp_audio_file.close()
            
            # Verify the audio file is valid
            try:
                mpe.AudioFileClip(temp_audio_file.name)
                return temp_audio_file.name
            except:
                os.unlink(temp_audio_file.name)
                logger.error("Downloaded file is not a valid audio file")
                return None
                
        logger.error(f"Failed to download audio: Status {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Error selecting background music: {str(e)}")
        return None

def generate_storyboard(prompt, style="motivational"):
    try:
        # Load the system prompt from the prompts file
        from prompts.autovideo_prompt import SYSTEM_PROMPT
        
        input_text = f"""Based on the video prompt: "{prompt}" and style: "{style}", 
        generate a detailed video storyboard in JSON format with this structure:
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
                    }},
                    "music_mood": "Suggested music mood for the scene",
                    "camera_movement": "Description of camera movement if any"
                }}
            ],
            "target_audience": "Description of the target audience",
            "overall_tone": "Description of the overall tone of the video",
            "total_duration": "Estimated total duration in seconds",
            "music_style": "Suggested music style for the video"
        }}"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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

def create_enhanced_scene_clip(scene, style_config, duration=5, max_retries=3):
    """Optimized scene clip creation"""
    clip = None
    for attempt in range(max_retries):
        try:
            # Pre-calculate dimensions
            width, height = 1920, 1080
            color1, color2 = style_config["color_scheme"]
            
            # Create gradient in memory-efficient way
            gradient = create_gradient_background(width, height, color1, color2)
            
            # Create base clip with optimized settings
            clip = mpe.ImageClip(gradient)
            clip = clip.set_duration(duration)
            
            # Process text elements in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_title = executor.submit(
                    create_animated_text,
                    scene['title'],
                    fontsize=70,
                    position=('center', 0.3)
                )
                future_desc = executor.submit(
                    create_animated_text,
                    scene['description'][:97] + "..." if len(scene['description']) > 100 else scene['description'],
                    fontsize=30,
                    position=('center', 0.6)
                )
                
                title_clip = future_title.result()
                desc_clip = future_desc.result()
            
            # Combine clips efficiently
            if title_clip and desc_clip:
                final_clip = mpe.CompositeVideoClip([clip, title_clip, desc_clip])
                return final_clip
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return create_fallback_clip(duration)
            time.sleep(1)
    
    return create_fallback_clip(duration)

def create_gradient_background(width, height, color1, color2):
    """Create a smooth gradient background."""
    # Convert hex colors to RGB
    c1 = np.array(tuple(int(color1[i:i+2], 16) for i in (1, 3, 5)))
    c2 = np.array(tuple(int(color2[i:i+2], 16) for i in (1, 3, 5)))
    
    gradient = np.linspace(0, 1, width)[:, np.newaxis]
    background = (c1 * (1 - gradient) + c2 * gradient).astype(np.uint8)
    return np.tile(background, (height, 1, 1))

def create_animated_text(text, fontsize=30, color='white', font='Arial', duration=5, position='center'):
    """Create text with enhanced animation effects."""
    txt_clip = mpe.TextClip(
        text,
        fontsize=fontsize,
        color=color,
        font=font,
        stroke_color='black',
        stroke_width=1
    )
    
    if isinstance(position, tuple):
        txt_clip = txt_clip.set_position(position)
    else:
        txt_clip = txt_clip.set_position(position)
    
    # Add fade in and subtle float animation
    txt_clip = (txt_clip
                .set_start(0)
                .crossfadein(0.5)
                .set_position(lambda t: (
                    position[0] if isinstance(position, tuple) else position,
                    position[1] + np.sin(t * 2) * 5 if isinstance(position, tuple) else 'center'
                )))
    
    return txt_clip.set_duration(duration)

def apply_transition(clip, transition_type):
    """Apply specified transition effect to the clip."""
    duration = clip.duration
    if transition_type == "fade":
        return clip.fadein(0.5).fadeout(0.5)
    elif transition_type == "slide":
        return clip.set_position(lambda t: ('center', 50*t))
    elif transition_type == "zoom":
        return clip.resize(lambda t: 1 + 0.1*t)
    return clip

def generate_voiceover(narration_text):
    logger.info(f"Generating voiceover for text: {narration_text[:50]}...")
    try:
        tts = gTTS(text=narration_text, lang='en', slow=False)
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_audio_file.name = os.path.join(tempfile.gettempdir(), f"voiceover_{int(time.time())}.mp3")
        tts.save(temp_audio_file.name)
        return mpe.AudioFileClip(temp_audio_file.name), temp_audio_file.name
    except Exception as e:
        logger.error(f"Error generating voiceover: {str(e)}")
        return mpe.AudioClip(lambda t: 0, duration=len(narration_text.split()) / 2), None

@contextmanager
def enhanced_cleanup_context():
    """Enhanced context manager for cleaning up temporary files and memory"""
    temp_files = []
    try:
        yield temp_files
    finally:
        # Clean up files
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.unlink(file)
            except Exception as e:
                logger.error(f"Failed to cleanup {file}: {e}")
        
        # Force garbage collection
        gc.collect()

def create_video(storyboard, background_music_file):
    with enhanced_cleanup_context() as temp_files:
        try:
            clips = []
            audio_clips = []
            
            # Process clips with proper resource management
            for scene in storyboard['scenes']:
                try:
                    clip = create_enhanced_scene_clip(
                        scene, 
                        get_cached_style_config(st.session_state.video_style),
                        duration=float(scene.get('duration', 5))
                    )
                    
                    if clip is None:
                        raise ValueError(f"Failed to create clip for scene {scene.get('scene_number')}")
                        
                    narration, temp_file = generate_voiceover(scene['narration'])
                    if temp_file:
                        temp_files.append(temp_file)
                        audio_clips.append(narration)
                    
                    clip = clip.set_audio(narration)
                    clips.append(clip)
                    
                except Exception as e:
                    logger.error(f"Error processing scene {scene.get('scene_number')}: {str(e)}")
                    # Use fallback clip instead of failing entirely
                    clips.append(create_fallback_clip(float(scene.get('duration', 5))))
            
            if not clips:
                raise ValueError("No valid clips were created")
                
            final_clip = mpe.concatenate_videoclips(clips)
            
            if background_music_file and os.path.exists(background_music_file):
                try:
                    background_music = mpe.AudioFileClip(background_music_file)
                    if background_music.duration < final_clip.duration:
                        num_loops = int(np.ceil(final_clip.duration / background_music.duration))
                        background_music = mpe.concatenate_audioclips([background_music] * num_loops)
                    background_music = background_music.subclip(0, final_clip.duration).volumex(0.1)
                    final_audio = mpe.CompositeAudioClip([final_clip.audio, background_music])
                    final_clip = final_clip.set_audio(final_audio)
                except Exception as e:
                    logger.error(f"Error processing background music: {str(e)}")
                    # Continue without background music rather than failing
            
            output_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.mp4',
                dir=tempfile.gettempdir()
            ).name
            temp_files.append(output_file)
            
            # Add error handling for video writing
            try:
                final_clip.write_videofile(
                    output_file, 
                    codec='libx264', 
                    audio_codec='aac', 
                    fps=24,
                    logger=None  # Suppress moviepy logging
                )
                return output_file
            except Exception as e:
                logger.error(f"Error writing video file: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error in create_video: {str(e)}")
            return None
        finally:
            # Explicitly close clips to free resources
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
            for audio in audio_clips:
                try:
                    audio.close()
                except:
                    pass

def apply_camera_movement(clip, movement_type):
    """Apply camera movement effects to the clip."""
    if movement_type == "zoom_in":
        return clip.resize(lambda t: 1 + (0.2 * t))
    elif movement_type == "zoom_out":
        return clip.resize(lambda t: 1.2 - (0.2 * t))
    elif movement_type == "pan_right":
        return clip.set_position(lambda t: (t * -100, 'center'))
    elif movement_type == "pan_left":
        return clip.set_position(lambda t: (t * 100, 'center'))
    return clip

def create_fallback_clip(duration=5):
    """Create a simple fallback clip when main clip creation fails."""
    try:
        # Create a simple black background with white text
        txt_clip = mpe.TextClip(
            "Scene content unavailable",
            fontsize=70,
            color='white',
            font='Arial-Bold',
            size=(1920, 1080)
        ).set_duration(duration)
        return txt_clip
    except Exception as e:
        logger.error(f"Fallback clip creation failed: {str(e)}")
        # Return absolute minimum clip - black screen
        return mpe.ColorClip(size=(1920, 1080), color=(0,0,0)).set_duration(duration)

def check_disk_space(required_mb=500):
    """Check if there's enough disk space available."""
    try:
        total, used, free = shutil.disk_usage(tempfile.gettempdir())
        free_mb = free // (2**20)  # Convert to MB
        if free_mb < required_mb:
            logger.error(f"Insufficient disk space. Required: {required_mb}MB, Available: {free_mb}MB")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        return False

def validate_storyboard(storyboard):
    """Validate storyboard structure and content."""
    try:
        required_fields = ['title', 'scenes', 'total_duration']
        if not all(field in storyboard for field in required_fields):
            return False
            
        if not storyboard['scenes'] or not isinstance(storyboard['scenes'], list):
            return False
            
        total_duration = float(storyboard['total_duration'])
        if total_duration > 300:  # Max 5 minutes
            return False
            
        for scene in storyboard['scenes']:
            if not all(field in scene for field in ['title', 'description', 'narration', 'duration']):
                return False
            if float(scene['duration']) > 60:  # Max 1 minute per scene
                return False
                
        return True
    except Exception as e:
        logger.error(f"Storyboard validation error: {str(e)}")
        return False

def verify_environment():
    """Verify all required components are available"""
    issues = []
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OpenAI API key not found")
    
    # Verify dependencies
    required_packages = ['moviepy', 'gtts', 'pydub']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Required package {package} not installed")
    
    return issues

def validate_prompt(prompt):
    """Validate user prompt"""
    if len(prompt) < 10:
        return False, "Prompt is too short"
    if len(prompt) > 500:
        return False, "Prompt is too long (max 500 characters)"
    
    # Basic content moderation
    forbidden_words = ['inappropriate', 'offensive', 'explicit']
    if any(word in prompt.lower() for word in forbidden_words):
        return False, "Prompt contains inappropriate content"
    
    return True, ""

def generate_storyboard_with_retry(prompt, style, max_retries=3):
    """Generate storyboard with retry mechanism"""
    for attempt in range(max_retries):
        try:
            with st.spinner(f'Attempt {attempt + 1}/{max_retries}: Generating storyboard...'):
                storyboard = generate_storyboard(prompt, style)
                if storyboard:
                    return storyboard
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retry
    return None

def estimate_processing_time(storyboard):
    """Estimate video processing time"""
    total_duration = float(storyboard['total_duration'])
    scene_count = len(storyboard['scenes'])
    
    # Rough estimate: 30 seconds per scene plus 2 min per minute of video
    estimate = (scene_count * 30) + (total_duration * 2)
    return estimate

def create_video_with_progress(storyboard, background_music_file):
    """Create video with progress updates"""
    try:
        total_scenes = len(storyboard['scenes'])
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for i, scene in enumerate(storyboard['scenes'], 1):
            progress_text.text(f"Processing scene {i}/{total_scenes}")
            progress_bar.progress(i/total_scenes)
            # Process scene...
            
        return final_video_path
    except Exception as e:
        logger.error(f"Video creation failed: {e}")
        return None

def handle_video_download(video_path):
    """Handle video download with cleanup"""
    try:
        file_size = os.path.getsize(video_path)
        if file_size > 200 * 1024 * 1024:  # 200MB
            st.warning("Large file size may cause download issues")
            
        with open(video_path, "rb") as file:
            st.download_button(
                label="Download Video",
                data=file,
                file_name="generated_video.mp4",
                mime="video/mp4",
                on_click=lambda: cleanup_temp_files([video_path])
            )
    except Exception as e:
        st.error(f"Download failed: {str(e)}")

class VideoProgress:
    """Progress tracking for video generation"""
    def __init__(self):
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.time_remaining = st.empty()
        self.start_time = time.time()
    
    def update(self, current, total, stage="Processing"):
        progress = current / total
        self.progress_bar.progress(progress)
        
        elapsed = time.time() - self.start_time
        estimated_total = elapsed / progress if progress > 0 else 0
        remaining = estimated_total - elapsed
        
        self.status_text.text(f"{stage}: {current}/{total}")
        self.time_remaining.text(f"Estimated time remaining: {remaining:.1f}s")
    
    def complete(self):
        self.progress_bar.progress(1.0)
        self.status_text.text("Complete!")
        self.time_remaining.empty()

class VideoGenerationError(Exception):
    """Custom exception for video generation errors"""
    pass

def create_video_with_recovery(storyboard, background_music_file):
    """Enhanced video creation with error recovery"""
    progress = VideoProgress()
    
    try:
        with enhanced_cleanup_context() as temp_files:
            clips = []
            total_scenes = len(storyboard['scenes'])
            
            for i, scene in enumerate(storyboard['scenes'], 1):
                try:
                    # Get cached style config
                    style_config = get_cached_style_config(st.session_state.video_style)
                    
                    # Create scene with progress tracking
                    progress.update(i, total_scenes, "Creating scene")
                    
                    clip = create_enhanced_scene_clip(
                        scene,
                        style_config,
                        duration=float(scene.get('duration', 5))
                    )
                    
                    if clip is None:
                        raise VideoGenerationError(f"Failed to create scene {i}")
                    
                    clips.append(clip)
                    
                except Exception as e:
                    logger.error(f"Scene {i} failed: {str(e)}")
                    clips.append(create_fallback_clip(float(scene.get('duration', 5))))
            
            # Combine clips with progress tracking
            progress.update(total_scenes, total_scenes, "Combining scenes")
            final_clip = mpe.concatenate_videoclips(clips)
            
            # Add background music if available
            if background_music_file:
                progress.update(total_scenes + 1, total_scenes + 2, "Adding music")
                final_clip = add_background_music(final_clip, background_music_file)
            
            # Write final video
            progress.update(total_scenes + 2, total_scenes + 2, "Saving video")
            output_file = write_video_file(final_clip)
            
            progress.complete()
            return output_file
            
    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}")
        st.error("Failed to generate video. Please try again.")
        return None

def write_video_file(clip, codec='libx264', audio_codec='aac', fps=24):
    """Optimized video file writing"""
    try:
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.mp4',
            dir=tempfile.gettempdir()
        ).name
        
        # Use optimized encoding settings
        clip.write_videofile(
            output_file,
            codec=codec,
            audio_codec=audio_codec,
            fps=fps,
            preset='faster',  # Faster encoding
            threads=2,  # Use multiple threads
            logger=None  # Suppress logging
        )
        
        return output_file
    except Exception as e:
        logger.error(f"Failed to write video file: {str(e)}")
        return None

def main():
    issues = verify_environment()
    if issues:
        st.error("Setup Issues Detected:")
        for issue in issues:
            st.warning(issue)
        return
        
    if not check_disk_space():
        st.error("Insufficient disk space available. Please free up at least 500MB.")
        return

    st.title("🎥 AutovideoAI")
    st.subheader("Create Professional Videos with AI")
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.update({
            'current_step': 1,
            'storyboard': None,
            'video_style': 'Motivational',
            'voice_option': list(VOICE_OPTIONS.keys())[0],
            'music_style': 'Electronic',
            'temp_files': set(),  # Track temporary files
            'last_prompt': None,  # Track last used prompt
            'processing_error': None  # Track processing errors
        })
    
    # Sidebar for settings and progress
    with st.sidebar:
        st.header("Progress")
        progress_bar = st.progress(0)
        st.markdown("---")
        st.subheader("Settings")
        video_style = st.selectbox(
            "Select Video Style",
            options=list(VIDEO_STYLES.keys()),
            help="Choose the overall style for your video"
        )
        
        voice_option = st.selectbox(
            "Select Voice",
            options=list(VOICE_OPTIONS.keys()),
            format_func=lambda x: VOICE_OPTIONS[x],
            help="Choose the narration voice"
        )

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Step 1: Input
        if st.session_state.current_step == 1:
            st.subheader("1. Enter Your Video Prompt")
            sample_prompts = load_sample_prompts()
            selected_prompt = st.selectbox(
                "Or choose a sample prompt:",
                options=[""] + sample_prompts[video_style],
                key="sample_prompt"
            )
            
            prompt = st.text_area(
                "Enter your video prompt:",
                value=selected_prompt,
                height=100,
                placeholder="Describe the video you want to create..."
            )
            
            if st.button("Generate Storyboard", type="primary"):
                if prompt:
                    with st.spinner("Generating storyboard..."):
                        st.session_state.storyboard = generate_storyboard_with_retry(prompt, video_style.lower())
                        if st.session_state.storyboard:
                            if not validate_storyboard(st.session_state.storyboard):
                                st.error("Invalid storyboard format or content")
                                st.session_state.current_step = 1
                                return
                            st.session_state.current_step = 2
                            progress_bar.progress(0.33)
                else:
                    st.error("Please enter a prompt first!")

        # Step 2: Review and Customize
        elif st.session_state.current_step == 2:
            st.subheader("2. Review and Customize")
            if st.session_state.storyboard:
                st.json(st.session_state.storyboard)
                
                if st.button("Generate Video", type="primary"):
                    st.session_state.current_step = 3
                    progress_bar.progress(0.66)
                
                if st.button("Back to Prompt", type="secondary"):
                    st.session_state.current_step = 1
                    progress_bar.progress(0)

        # Step 3: Generate and Download
        elif st.session_state.current_step == 3:
            st.subheader("3. Generate Video")
            with st.spinner("Creating your video..."):
                background_music = select_background_music(
                    st.session_state.get('music_style', 'Electronic')
                )
                video_file = create_video(
                    st.session_state.storyboard,
                    background_music
                )
                
                if video_file:
                    st.success("Video created successfully!")
                    st.video(video_file)
                    progress_bar.progress(1.0)
                    
                    handle_video_download(video_file)
                    
                    if st.button("Create Another Video", type="primary"):
                        st.session_state.current_step = 1
                        progress_bar.progress(0)
                else:
                    st.error("Failed to generate video. Please try again.")
                    if st.button("Try Again", type="primary"):
                        st.session_state.current_step = 2
                        progress_bar.progress(0.33)

    with col2:
        st.subheader("Style Preview")
        st.markdown(f"**{video_style}**")
        st.markdown(VIDEO_STYLES[video_style]["description"])
        st.markdown("---")
        st.markdown("### Features")
        st.markdown("- Professional transitions")
        st.markdown("- Custom color scheme")
        st.markdown("- AI-powered narration")
        st.markdown("- Royalty-free music")

if __name__ == "__main__":
    main()