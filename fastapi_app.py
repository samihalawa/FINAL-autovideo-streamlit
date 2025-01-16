from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
import os, logging, json, time, shutil, random, gc
import moviepy.editor as mpe
import numpy as np
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from functools import lru_cache
import concurrent.futures
from gtts import gTTS
from dotenv import load_dotenv
import requests

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "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")
HUGGINGFACE_API_BASE = os.getenv("HUGGINGFACE_API_BASE", "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# Output directory setup
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = BASE_DIR / "video_outputs"
TEMP_DIR = OUTPUT_DIR / "temp"
FINAL_DIR = OUTPUT_DIR / "final"

# Create output directories with full permissions
OUTPUT_DIR.mkdir(mode=0o777, exist_ok=True)
TEMP_DIR.mkdir(mode=0o777, exist_ok=True)
FINAL_DIR.mkdir(mode=0o777, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="AutovideoAI API",
    debug=True,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with absolute path
app.mount("/video_outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="video_outputs")

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

# Pydantic models
class VideoRequest(BaseModel):
    prompt: str
    style: str = "Motivational"
    music_style: str = "Electronic"
    voice_option: str = "en-US-Standard"

class VideoResponse(BaseModel):
    video_path: str
    duration: float
    style: str
    status: str

def generate_storyboard(prompt: str, style: str = "motivational") -> dict:
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        system_prompt = """You are a creative video storyboard generator. Create a video storyboard with engaging scenes, descriptions, and narrations."""
        
        user_prompt = f"""Create a video storyboard about {prompt} in {style} style.
        Format the response as a JSON with the following structure:
        {{
            "title": "Video Title",
            "scenes": [
                {{
                    "title": "Scene Title",
                    "description": "Scene Description",
                    "narration": "Scene Narration",
                    "duration": 5.0
                }}
            ]
        }}
        Make it creative and engaging."""
        
        payload = {
            "inputs": f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant: Let me create a storyboard for you.",
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        response = requests.post(HUGGINGFACE_API_BASE, headers=headers, json=payload)
        response.raise_for_status()
        
        generated_text = response.json()[0]["generated_text"]
        try:
            # Try to find JSON-like content in the response
            import re
            json_match = re.search(r'({[\s\S]*})', generated_text)
            if json_match:
                storyboard = json.loads(json_match.group(1))
            else:
                raise ValueError("No JSON found in response")
        except:
            # Fallback structure
            storyboard = {
                "title": prompt,
                "scenes": [
                    {
                        "title": "Opening Scene",
                        "description": f"Introduction to {prompt}",
                        "narration": f"Welcome to our video about {prompt}.",
                        "duration": 5.0
                    },
                    {
                        "title": "Main Content",
                        "description": f"Exploring {prompt}",
                        "narration": f"Let's explore {prompt} in detail.",
                        "duration": 10.0
                    },
                    {
                        "title": "Closing",
                        "description": "Conclusion",
                        "narration": "Thank you for watching!",
                        "duration": 5.0
                    }
                ]
            }
        
        return storyboard
        
    except Exception as e:
        logger.error(f"Storyboard generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_unique_filename(base_dir: Path, prefix: str, suffix: str) -> Path:
    """Generate a unique filename in the specified directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices('0123456789', k=4))
    return base_dir / f"{prefix}_{timestamp}_{random_suffix}{suffix}"

def create_session_dir() -> Path:
    """Create a new session directory with timestamp"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    return session_dir

def create_enhanced_scene_clip(scene: dict, style_config: dict, duration: float = 5) -> mpe.VideoClip:
    try:
        width, height = 1920, 1080
        color1, color2 = style_config["color_scheme"]
        
        # Create solid color background
        color_clip = mpe.ColorClip(size=(width, height), color=color1[1:])
        clip = color_clip.set_duration(duration)
        
        # Add text overlays using simpler text method
        title_text = mpe.TextClip(
            scene['title'], 
            fontsize=70, 
            color='white',
            bg_color='transparent',
            font='Arial',
            method='label'
        ).set_position(('center', height//3)).set_duration(duration)
        
        desc_text = mpe.TextClip(
            scene['description'], 
            fontsize=30, 
            color='white',
            bg_color='transparent',
            font='Arial',
            method='label'
        ).set_position(('center', height//2)).set_duration(duration)
        
        return mpe.CompositeVideoClip([clip, title_text, desc_text])
        
    except Exception as e:
        logger.error(f"Scene clip creation error: {e}")
        # Return a simple black clip if text overlay fails
        return mpe.ColorClip(size=(1920, 1080), color=(0,0,0)).set_duration(duration)

def generate_voiceover(text: str, scene_index: int = 0) -> Optional[str]:
    try:
        tts = gTTS(text=text, lang='en')
        voice_file = get_unique_filename(TEMP_DIR, f"voice_{scene_index:02d}", ".mp3")
        tts.save(voice_file)
        return voice_file
    except Exception as e:
        logger.error(f"Voiceover error: {e}")
        return None

def select_background_music(genre: str) -> Optional[str]:
    try:
        track_url = MUSIC_TRACKS.get(genre, random.choice(list(MUSIC_TRACKS.values())))
        response = requests.get(track_url, timeout=10)
        
        if response.status_code != 200 or 'audio' not in response.headers.get('content-type', ''):
            logger.error(f"Invalid music track response for {genre}")
            return None
            
        music_file = get_unique_filename(TEMP_DIR, "background_music", ".mp3")
        with open(music_file, 'wb') as f:
            f.write(response.content)
        return music_file
        
    except Exception as e:
        logger.error(f"Error selecting music: {e}")
        return None

def create_video(storyboard: dict, background_music_file: str, session_dir: Path) -> str:
    try:
        if not storyboard or 'scenes' not in storyboard:
            raise ValueError("Invalid storyboard format")
        
        clips = []
        scene_files = []
        voice_files = []
        
        # Process each scene
        for i, scene in enumerate(storyboard['scenes']):
            # Create scene video
            scene_file = session_dir / f"scene_{i:02d}.mp4"
            clip = create_enhanced_scene_clip(scene, VIDEO_STYLES[scene.get('style', 'Motivational')], float(scene.get('duration', 5)))
            
            # Save scene video
            clip.write_videofile(str(scene_file), codec='libx264', audio_codec='aac', fps=24, logger=None)
            scene_files.append(scene_file)
            
            # Create voiceover
            voice_file = session_dir / f"voice_{i:02d}.mp3"
            tts = gTTS(text=scene['narration'], lang='en')
            tts.save(str(voice_file))
            voice_files.append(voice_file)
            
            # Add voiceover to clip
            narration = mpe.AudioFileClip(str(voice_file))
            clip = clip.set_audio(narration)
            clips.append(clip)
        
        # Combine all clips
        final_clip = mpe.concatenate_videoclips(clips)
        
        # Add background music if provided
        if background_music_file:
            bg_music = mpe.AudioFileClip(background_music_file).volumex(0.1)
            music_duration = bg_music.duration
            video_duration = final_clip.duration
            loop_times = int(np.ceil(video_duration / music_duration))
            
            bg_music_loops = [bg_music] * loop_times
            bg_music_combined = mpe.concatenate_audioclips(bg_music_loops)
            bg_music_combined = bg_music_combined.subclip(0, video_duration)
            
            final_clip = final_clip.set_audio(
                mpe.CompositeAudioClip([final_clip.audio, bg_music_combined])
            )
        
        # Save final video
        final_video_path = get_unique_filename(FINAL_DIR, "final_video", ".mp4")
        final_clip.write_videofile(str(final_video_path), codec='libx264', audio_codec='aac', fps=24, logger=None)
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "scenes": len(scene_files),
            "duration": final_clip.duration,
            "has_background_music": background_music_file is not None,
            "scene_files": [str(f.relative_to(OUTPUT_DIR)) for f in scene_files],
            "voice_files": [str(f.relative_to(OUTPUT_DIR)) for f in voice_files],
            "final_video": str(final_video_path.relative_to(OUTPUT_DIR))
        }
        
        metadata_file = session_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(final_video_path)
        
    except Exception as e:
        logger.error(f"Video creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-video", response_model=VideoResponse)
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    try:
        # Create session directory
        session_dir = create_session_dir()
        logger.info(f"Created session directory: {session_dir}")
        
        # Generate storyboard
        storyboard = generate_storyboard(request.prompt, request.style.lower())
        
        # Save storyboard JSON
        storyboard_file = session_dir / "storyboard.json"
        with open(storyboard_file, 'w') as f:
            json.dump(storyboard, f, indent=2)
        logger.info(f"Saved storyboard to: {storyboard_file}")
        
        # Select background music
        bg_music_file = select_background_music(request.music_style)
        if bg_music_file:
            # Move background music to session directory
            new_music_file = session_dir / os.path.basename(bg_music_file)
            shutil.move(bg_music_file, str(new_music_file))
            bg_music_file = str(new_music_file)
            logger.info(f"Moved background music to: {new_music_file}")
        
        # Create video
        video_path = create_video(storyboard, bg_music_file, session_dir)
        logger.info(f"Created final video at: {video_path}")
        
        # Calculate duration
        duration = sum(float(scene.get('duration', 5)) for scene in storyboard['scenes'])
        
        # Create relative path for response
        relative_path = os.path.relpath(video_path, OUTPUT_DIR)
        
        return VideoResponse(
            video_path=f"/video_outputs/{relative_path}",
            duration=duration,
            style=request.style,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    
    # Print startup message
    print("Starting AutoVideo API server...")
    print("API Documentation available at: http://localhost:8088/docs")
    print("Health check endpoint: http://localhost:8088/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8088, log_level="debug")

# Requirements (add these to requirements.txt):
"""
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
requests==2.31.0
moviepy==1.0.3
numpy==1.24.3
gTTS==2.3.2
pydantic==2.4.2
python-multipart==0.0.6
""" 