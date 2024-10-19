import streamlit as st
from openai import OpenAI
import os
import moviepy.editor as mpe
import requests
from tempfile import NamedTemporaryFile
from moviepy.video.fx.all import fadein, fadeout, resize
import psutil
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
from PIL import Image, ImageDraw, ImageFont
import shutil
import logging
from gtts import gTTS
from dotenv import load_dotenv
from pydub import AudioSegment
import random
import moviepy.video.fx.all as vfx
import cv2
from pydub.silence import split_on_silence
import numpy as np
import tempfile
from functools import lru_cache
from huggingface_hub import InferenceClient, hf_hub_download
from huggingface_hub import login, HfApi
from datasets import load_dataset
import textwrap
from scenedetect import detect, ContentDetector
from pydub import AudioSegment
from pydub.silence import split_on_silence
import re
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm
import pandas as pd
import cv2
from sentence_transformers import SentenceTransformer
import torch
from yt_dlp import YoutubeDL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config at the very beginning of the script
st.set_page_config(page_title="AutovideoAI", page_icon="🎥", layout="wide")

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Check if already logged in
api = HfApi()
try:
    api.whoami(token=hf_token)
    logger.info("Already logged in to Hugging Face")
except Exception:
    logger.info("Logging in to Hugging Face")
    login(token=hf_token, add_to_git_credential=False)

# Update OpenAI client initialization
client = OpenAI()

# Theme customization
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Apply theme
if st.session_state.theme == 'dark':
    st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

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

# Set up caching for downloaded assets
@lru_cache(maxsize=100)
def cached_download(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    return None

# 1. Function to generate storyboard based on user prompt using structured JSON
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
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
            model="gpt-4o-mini",
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

def validate_storyboard(storyboard):
    if "title" not in storyboard or "scenes" not in storyboard or not isinstance(storyboard["scenes"], list):
        return False
    if len(storyboard["scenes"]) < 3:
        return False
    
    required_fields = ["scene_number", "title", "description", "narration", "keywords", "duration", "overlay_text", "visual_elements", "audio_cues", "transitions"]
    return all(all(field in scene for field in required_fields) for scene in storyboard["scenes"])

# 2. Function to parse structured JSON storyboard data
def parse_storyboard(storyboard):
    try:
        return json.loads(storyboard).get("scenes", [])
    except json.JSONDecodeError:
        return []

# 3. Function to fetch video clips dynamically based on scene keywords
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

@st.cache_resource
def load_video_dataset():
    return load_dataset("yttemporal1b", split="train", streaming=True)

def fetch_video_clips_optimized(scenes):
    logger.info(f"Fetching video clips for {len(scenes)} scenes")
    video_clips = []
    
    model = load_sentence_transformer()
    dataset = load_video_dataset()
    
    for i, scene in enumerate(scenes):
        logger.info(f"Fetching clip for scene {i+1}: {scene['title']}")
        
        query = f"{scene['title']} {scene['description']}"
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        best_score = -1
        best_video = None
        
        for batch in dataset.iter(batch_size=100):
            batch_embeddings = model.encode(batch['title'], convert_to_tensor=True)
            cos_scores = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), batch_embeddings)
            max_score, max_index = torch.max(cos_scores, dim=0)
            
            if max_score > best_score:
                best_score = max_score
                best_video = batch['url'][max_index]
            
            if best_score > 0.8:  # Early stopping if we find a good match
                break
        
        if best_video:
            try:
                with YoutubeDL({'format': 'best[height<=720]'}) as ydl:
                    info = ydl.extract_info(best_video, download=False)
                    video_url = info['url']
                
                clip = mpe.VideoFileClip(video_url, audio=False).subclip(0, min(float(scene['duration']), 10))
                clip = clip.resize(height=720).set_fps(30)
                video_clips.append({'clip': clip, 'scene': scene})
            except Exception as e:
                logger.warning(f"Error processing video for scene {i+1}: {str(e)}")
        
        if len(video_clips) <= i:
            logger.warning(f"No suitable video found for scene {i+1}. Creating fallback clip.")
            clip = create_fallback_clip(scene, duration=min(float(scene['duration']), 10))
            video_clips.append({'clip': clip, 'scene': scene})
    
    return video_clips

def create_fallback_clip(scene, duration=5):
    text = scene.get('title', 'Scene')
    size = (1280, 720)
    
    img = Image.new('RGB', size, color='black')
    draw = ImageDraw.Draw(img)
    
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    
    wrapped_text = textwrap.wrap(text, width=30)
    y_text = (size[1] - len(wrapped_text) * 50) // 2
    
    for line in wrapped_text:
        line_width, line_height = draw.textsize(line, font=font)
        position = ((size[0] - line_width) / 2, y_text)
        draw.text(position, line, font=font, fill='white')
        y_text += line_height + 10
    
    img_array = np.array(img)
    clip = mpe.ImageClip(img_array).set_duration(duration)
    return clip.set_fps(30)

# 4. Function to generate voiceover with Hugging Face Inference API
def generate_voiceover(narration_text):
    logger.info(f"Generating voiceover for text: {narration_text[:50]}...")
    try:
        tts = gTTS(text=narration_text, lang='en', slow=False)
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_audio_file.name)
        return mpe.AudioFileClip(temp_audio_file.name)
    except Exception as e:
        logger.error(f"Error generating voiceover: {str(e)}")
        return create_silent_audio(len(narration_text.split()) / 2)  # Assuming 2 words per second

def create_silent_audio(duration):
    silent_segment = AudioSegment.silent(duration=int(duration * 1000))  # pydub uses milliseconds
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    silent_segment.export(temp_audio_file.name, format="mp3")
    return mpe.AudioFileClip(temp_audio_file.name)

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

# Add this function for smooth transitions
def add_fade_transition(clip1, clip2, duration=1):
    return mpe.CompositeVideoClip([clip1.crossfadeout(duration), clip2.crossfadein(duration)])

# Add this function for dynamic text animations
def create_animated_text(text, duration=5, font_size=70, color='white'):
    try:
        # Create a black background image
        img = Image.new('RGB', (1280, 720), color='black')
        draw = ImageDraw.Draw(img)
        
        # Use a default font
        font = ImageFont.load_default().font_variant(size=font_size)
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate position to center the text
        position = ((1280 - text_width) / 2, (720 - text_height) / 2)
        
        # Draw the text
        draw.text(position, text, font=font, fill=color)
        
        # Convert to numpy array and create video clip
        img_array = np.array(img)
        clip = mpe.ImageClip(img_array).set_duration(duration)
        
        # Add fade in and fade out effects
        clip = clip.fadein(1).fadeout(1)
        
        return clip
    except Exception as e:
        logger.error(f"Error creating animated text: {e}")
        return mpe.ColorClip(size=(1280, 720), color=(0,0,0)).set_duration(duration)

# Add this function for color grading
def apply_color_grading(clip, brightness=1.0, contrast=1.0, saturation=1.0):
    return clip.fx(vfx.colorx, brightness).fx(vfx.lum_contrast, contrast=contrast).fx(vfx.colorx, saturation)

# Add this function for creating lower thirds
def create_lower_third(text, duration):
    try:
        # Create a transparent background
        img = Image.new('RGBA', (1280, 720), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Use a default font
        font = ImageFont.load_default().font_variant(size=30)
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate position for lower third
        position = ((1280 - text_width) / 2, 720 - text_height - 50)
        
        # Draw semi-transparent background
        bg_bbox = (position[0]-10, position[1]-10, position[0]+text_width+10, position[1]+text_height+10)
        draw.rectangle(bg_bbox, fill=(0,0,0,153))
        
        # Draw the text
        draw.text(position, text, font=font, fill='white')
        
        # Convert to numpy array and create video clip
        img_array = np.array(img)
        return mpe.ImageClip(img_array).set_duration(duration)
    except Exception as e:
        logger.error(f"Error creating lower third: {e}")
        return mpe.ColorClip(size=(1280, 720), color=(0,0,0,0)).set_duration(duration)

def smart_cut(video_clip, audio_clip):
    try:
        # Check if audio_clip is a file path or an AudioClip object
        if isinstance(audio_clip, str):
            audio_segment = AudioSegment.from_wav(audio_clip)
        elif isinstance(audio_clip, mpe.AudioClip):
            # Convert AudioClip to numpy array
            audio_array = audio_clip.to_soundarray()
            audio_segment = AudioSegment(
                audio_array.tobytes(),
                frame_rate=audio_clip.fps,
                sample_width=audio_array.dtype.itemsize,
                channels=1 if audio_array.ndim == 1 else audio_array.shape[1]
            )
        else:
            logger.warning("Unsupported audio type. Returning original video clip.")
            return video_clip

        # Split audio on silences
        chunks = split_on_silence(audio_segment, min_silence_len=500, silence_thresh=-40)
        
        # Calculate timestamps for cuts
        cut_times = [0]
        for chunk in chunks:
            cut_times.append(cut_times[-1] + len(chunk) / 1000)
        
        # Cut video based on audio
        cut_clips = [video_clip.subclip(start, end) for start, end in zip(cut_times[:-1], cut_times[1:])]
        
        return mpe.concatenate_videoclips(cut_clips)
    except Exception as e:
        logger.error(f"Error in smart_cut: {str(e)}")
        return video_clip

def apply_speed_changes(clip, speed_factor=1.5, threshold=0.1):
    try:
        if not hasattr(clip, 'fps') or clip.fps is None:
            clip = clip.set_fps(24)  # Set a default fps if not present
        
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in clip.iter_frames()]
        motion = [np.mean(cv2.absdiff(frames[i], frames[i+1])) for i in range(len(frames)-1)]
        
        speed_clip = clip.fl(lambda gf, t: gf(t * speed_factor) if motion[int(t*clip.fps)] < threshold else gf(t))
        
        return speed_clip
    except Exception as e:
        logger.error(f"Error in apply_speed_changes: {str(e)}")
        return clip

# 7. Function to create and finalize the video
def create_video(video_clips, background_music_file, video_title):
    logger.info(f"Starting video creation process for '{video_title}'")
    try:
        clips = []
        for i, clip_data in enumerate(video_clips):
            try:
                clip = clip_data['clip']
                narration = clip_data['narration']
                clip = clip.set_audio(narration)
                clips.append(clip)
                logger.info(f"Processed clip {i+1} successfully")
            except Exception as e:
                logger.error(f"Error processing clip {i+1}: {str(e)}")
        
        if not clips:
            raise ValueError("No valid clips were created")
        
        logger.info(f"Concatenating {len(clips)} scene clips")
        final_clip = mpe.concatenate_videoclips(clips)
        
        if background_music_file:
            logger.info("Adding background music")
            background_music = mpe.AudioFileClip(background_music_file).volumex(0.1)
            background_music = background_music.audio_loop(duration=final_clip.duration)
            final_audio = mpe.CompositeAudioClip([final_clip.audio, background_music])
            final_clip = final_clip.set_audio(final_audio)
        
        # Add intro and outro
        intro_clip = mpe.TextClip(video_title, fontsize=70, color='white', size=(1280, 720), bg_color='black').set_duration(3)
        outro_clip = mpe.TextClip("Thanks for watching!", fontsize=70, color='white', size=(1280, 720), bg_color='black').set_duration(3)
        final_clip = mpe.concatenate_videoclips([intro_clip, final_clip, outro_clip])
        
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        logger.info(f"Writing final video to {output_file}")
        final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac', fps=24)
        logger.info("Video creation process completed")
        return output_file
    except Exception as e:
        logger.error(f"Error in create_video: {str(e)}")
        st.error(f"An error occurred while creating the video: {str(e)}")
        return None

def enhance_clip(clip, script_analysis):
    # Apply color grading based on sentiment
    if script_analysis['sentiment'] == 'POSITIVE':
        clip = apply_color_grading(clip, brightness=1.1, saturation=1.2)
    elif script_analysis['sentiment'] == 'NEGATIVE':
        clip = apply_color_grading(clip, brightness=0.9, contrast=1.1)
    
    # Add dynamic text animations
    if clip.duration > 2:
        text_clip = create_animated_text(clip.scene['title'], duration=2)
        clip = mpe.CompositeVideoClip([clip, text_clip.set_start(1)])
    
    # Add smooth transitions
    clip = clip.crossfadein(0.5).crossfadeout(0.5)
    
    return clip

def process_clip(enhanced_clip, clip_data, script_analysis):
    scene = clip_data['scene']
    duration = float(scene['duration'])
    processed_clip = enhanced_clip.set_duration(duration)
    
    # Apply color grading based on sentiment and style
    if script_analysis['sentiment'] == 'POSITIVE':
        processed_clip = apply_color_grading(processed_clip, brightness=1.1)
    else:
        processed_clip = apply_color_grading(processed_clip, brightness=0.9)
    
    if script_analysis['style'] in ['humorous', 'casual']:
        processed_clip = processed_clip.fx(vfx.colorx, 1.2)  # More vibrant for humorous/casual content
    elif script_analysis['style'] in ['dramatic', 'formal']:
        processed_clip = processed_clip.fx(vfx.lum_contrast, contrast=1.2)  # More contrast for dramatic/formal content
    
    if scene.get('overlay_text'):
        text_clip = create_animated_text(scene['overlay_text'], duration)
        processed_clip = mpe.CompositeVideoClip([processed_clip, text_clip])
    
    lower_third = create_lower_third(scene['title'], duration)
    processed_clip = mpe.CompositeVideoClip([processed_clip, lower_third])
    
    return processed_clip

# 8. Function to apply fade-in/fade-out effects to video clips
def apply_fade_effects(clip, duration=1):
    try:
        return fadein(clip, duration).fx(fadeout, duration)
    except Exception as e:
        raise ValueError(f"Error applying fade effects: {e}")

# 9. Function to add text overlay to video clips
def add_text_overlay(clip, text):
    if text:
        try:
            text_clip = mpe.TextClip(text, fontsize=70, color='white', font='Arial-Bold')
            text_clip = text_clip.set_position('center').set_duration(clip.duration)
            return mpe.CompositeVideoClip([clip, text_clip])
        except Exception as e:
            raise ValueError(f"Error adding text overlay: {e}")
    return clip

# 10. Function to add narration to video clip
def add_narration(clip, narration_file):
    try:
        return clip.set_audio(mpe.AudioFileClip(narration_file))
    except Exception as e:
        raise ValueError(f"Error adding narration: {e}")

# 11. Function to add background music to video
def add_background_music(clip, music_file):
    try:
        background_audio = mpe.AudioFileClip(music_file)
        return clip.set_audio(mpe.CompositeAudioClip([clip.audio, background_audio.volumex(0.1)]))
    except Exception as e:
        raise ValueError(f"Error adding background music: {e}")

# 12. Function to add watermarks to video clips
def add_watermark(clip, watermark_text="Sample Watermark"):
    try:
        watermark = mpe.TextClip(watermark_text, fontsize=30, color='white', font='Arial')
        watermark = watermark.set_position(('right', 'bottom')).set_duration(clip.duration)
        return mpe.CompositeVideoClip([clip, watermark])
    except Exception as e:
        st.error(f"Error adding watermark: {e}")
        return clip  # Return original clip if watermark fails

# 13. Function to split video into parts for processing
def split_video(video_clip, part_duration=10):
    try:
        return [video_clip.subclip(start, min(start + part_duration, video_clip.duration)) for start in range(0, int(video_clip.duration), part_duration)]
    except Exception as e:
        st.error(f"Error splitting video: {e}")
        return [video_clip]  # Return original clip if splitting fails

# 14. Function to merge video parts back together
def merge_video_parts(video_parts):
    try:
        return mpe.concatenate_videoclips(video_parts, method="compose")
    except Exception as e:
        st.error(f"Error merging video parts: {e}")
        return video_parts[0] if video_parts else None  # Return first part if merging fails

# 15. Function to save a temporary JSON backup of generated storyboard
def save_storyboard_backup(storyboard, filename="storyboard_backup.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(storyboard, f)
        st.success(f"Storyboard backup saved to {filename}")
    except Exception as e:
        st.error(f"Error saving storyboard backup: {e}")

# 16. Function to load a saved storyboard from backup
def load_storyboard_backup(filename="storyboard_backup.json"):
    try:
        with open(filename, 'r') as f:
            storyboard = json.load(f)
        st.success(f"Storyboard loaded from {filename}")
        return storyboard
    except FileNotFoundError:
        st.warning(f"Backup file {filename} not found.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filename}")
        return None
    except Exception as e:
        st.error(f"Error loading storyboard backup: {e}")
        return None

# 17. Function to add subtitles to video
def add_subtitles_to_video(clip, subtitles):
    try:
        subtitle_clips = [
            mpe.TextClip(subtitle['text'], fontsize=50, color='white', size=clip.size, font='Arial-Bold')
            .set_position(('bottom')).set_start(subtitle['start']).set_duration(subtitle['duration'])
            for subtitle in subtitles
        ]
        return mpe.CompositeVideoClip([clip] + subtitle_clips)
    except Exception as e:
        st.error(f"Error adding subtitles: {e}")
        return clip  # Return original clip if adding subtitles fails

# 18. Function to preview storyboard as a slideshow
def preview_storyboard_slideshow(scenes, duration_per_scene=5):
    try:
        slides = [create_animated_text(scene['title'], duration=duration_per_scene) for scene in scenes]
        slideshow = mpe.concatenate_videoclips(slides, method='compose')
        slideshow.write_videofile("storyboard_preview.mp4", codec='libx264')
        st.success("Storyboard preview created successfully.")
        st.video("storyboard_preview.mp4")
    except Exception as e:
        st.error(f"Error creating storyboard slideshow: {e}")

# 19. Function to add logo to video
def add_logo_to_video(clip, logo_path, position=('right', 'top')):
    try:
        logo = mpe.ImageClip(logo_path).set_duration(clip.duration).resize(height=100).set_position(position)
        return mpe.CompositeVideoClip([clip, logo])
    except FileNotFoundError:
        st.error(f"Logo file not found: {logo_path}")
        return clip
    except Exception as e:
        st.error(f"Error adding logo to video: {e}")
        return clip  # Return original clip if adding logo fails

# 20. Function to compress video output for faster uploading
def compress_video(input_path, output_path="compressed_video.mp4", bitrate="500k"):
    try:
        os.system(f"ffmpeg -i {input_path} -b:v {bitrate} -bufsize {bitrate} {output_path}")
        st.success(f"Video compressed successfully. Saved to {output_path}")
    except Exception as e:
        st.error(f"Error compressing video: {e}")

# 21. Function to apply black-and-white filter to video
def apply_bw_filter(clip):
    try:
        return clip.fx(mpe.vfx.blackwhite)
    except Exception as e:
        st.error(f"Error applying black-and-white filter: {e}")
        return clip  # Return original clip if filter fails

# 23. Function to overlay images on video
def overlay_image_on_video(clip, image_path, position=(0, 0)):
    try:
        image = mpe.ImageClip(image_path).set_duration(clip.duration).set_position(position)
        return mpe.CompositeVideoClip([clip, image])
    except FileNotFoundError:
        st.error(f"Image file not found: {image_path}")
        return clip
    except Exception as e:
        st.error(f"Error overlaying image on video: {e}")
        return clip  # Return original clip if overlay fails

# 24. Function to adjust video speed
def adjust_video_speed(clip, speed=1.0):
    try:
        return clip.fx(mpe.vfx.speedx, speed)
    except Exception as e:
        st.error(f"Error adjusting video speed: {e}")
        return clip  # Return original clip if speed adjustment fails

# 25. Function to crop video clips
def crop_video(clip, x1, y1, x2, y2):
    try:
        return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)
    except Exception as e:
        st.error(f"Error cropping video: {e}")
        return clip  # Return original clip if cropping fails

# 26. Function to adjust resolution dynamically based on system capacity
def adjust_resolution_based_on_system(clip):
    try:
        memory = psutil.virtual_memory()
        resolution = (640, 360) if memory.available < 1000 * 1024 * 1024 else (1280, 720)
        return resize(clip, newsize=resolution)
    except Exception as e:
        st.error(f"Error adjusting resolution: {e}")
        return clip  # Return original clip if resolution adjustment fails

# 27. Function to generate video thumbnail
def generate_video_thumbnail(clip, output_path="thumbnail.png"):
    try:
        frame = clip.get_frame(1)
        image = Image.fromarray(frame)
        image.save(output_path)
        st.success(f"Thumbnail generated successfully. Saved to {output_path}")
        return output_path
    except Exception as e:
        st.error(f"Error generating video thumbnail: {e}")
        return None

# 29. Function to add intro and outro sequences to video
def add_intro_outro(final_clip, video_title):
    intro_clip = create_animated_text(video_title, duration=3, font_size=60).fx(vfx.fadeout, duration=1)
    outro_clip = create_animated_text("Thanks for Watching!", duration=3, font_size=60).fx(vfx.fadein, duration=1)
    return mpe.concatenate_videoclips([intro_clip, final_clip, outro_clip])

# 30. Function to adjust audio volume levels
def adjust_audio_volume(audio_clip, volume_level=1.0):
    try:
        return audio_clip.volumex(volume_level)
    except Exception as e:
        st.error(f"Error adjusting audio volume: {e}")
        return audio_clip  # Return original audio clip if volume adjustment fails

# 31. Function to generate a text overlay with gradient background
def generate_gradient_text_overlay(text, clip_duration, size=(1920, 1080)):
    try:
        gradient = color_gradient(size, p1=(0, 0), p2=(size[0], size[1]), color1=(255, 0, 0), color2=(0, 0, 255))
        image = Image.fromarray(gradient)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text_size = draw.textsize(text, font=font)
        draw.text(((size[0] - text_size[0]) / 2, (size[1] - text_size[1]) / 2), text, font=font, fill=(255, 255, 255))
        image.save("gradient_overlay.png")
        return mpe.ImageClip("gradient_overlay.png").set_duration(clip_duration)
    except Exception as e:
        st.error(f"Error generating gradient text overlay: {e}")
        return mpe.TextClip(text, fontsize=70, color='white', size=size).set_duration(clip_duration)

# 32. Function to run video rendering in a separate thread
def run_video_rendering_thread(target_function, *args):
    try:
        rendering_thread = threading.Thread(target=target_function, args=args)
        rendering_thread.start()
        return rendering_thread
    except Exception as e:
        st.error(f"Error running rendering thread: {e}")
        return None

# 33. Function to check system capabilities before rendering
def check_system_capabilities():
    try:
        memory = psutil.virtual_memory()
        if memory.available < 500 * 1024 * 1024:  # Less than 500MB
            st.warning("Low memory detected. Consider closing other applications.")
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > 80:
            st.warning("High CPU usage detected. Rendering may be slow.")
    except Exception as e:
        st.error(f"Error checking system capabilities: {e}")

# 34. Function to log system resources during video generation
def log_system_resources():
    try:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        st.write(f"Memory Usage: {memory.percent}% | CPU Usage: {cpu}%")
    except Exception as e:
        st.error(f"Error logging system resources: {e}")

# 35. Function to download additional video assets (e.g., background music)
def download_additional_assets(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            temp_asset_file = NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_asset_file.write(response.content)
            temp_asset_file.flush()
            st.success(f"Asset downloaded successfully: {temp_asset_file.name}")
            return temp_asset_file.name
        else:
            st.error("Failed to download asset. Invalid URL or server error.")
            return None
    except Exception as e:
        st.error(f"Error downloading asset: {e}")
        return None

# 36. Function to calculate estimated video rendering time
def calculate_estimated_render_time(duration, resolution=(1280, 720)):
    try:
        estimated_time = duration * (resolution[0] * resolution[1]) / 1e6
        st.info(f"Estimated rendering time: {estimated_time:.2f} seconds")
        return estimated_time
    except Exception as e:
        st.error(f"Error calculating render time: {e}")
        return None

# 37. Function to manage temporary directories
def manage_temp_directory(directory_path):
    try:
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        os.makedirs(directory_path)
        st.success(f"Temporary directory created: {directory_path}")
    except Exception as e:
        st.error(f"Error managing temporary directory: {e}")

# 38. Function to handle session expiration or token errors
def handle_session_expiration():
    try:
        st.error("Session expired. Please refresh and try again.")
        if st.button("Refresh Page"):
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    except Exception as e:
        st.error(f"Error handling session expiration: {e}")

# 39. Function to split storyboard scenes for easy preview
def split_storyboard_scenes(scenes, batch_size=5):
    try:
        return [scenes[i:i + batch_size] for i in range(0, len(scenes), batch_size)]
    except Exception as e:
        st.error(f"Error splitting storyboard scenes: {e}")
        return [scenes]  # Return all scenes in one batch if splitting fails

# 40. Function to add transition effects between storyboard scenes
def add_transition_effects_between_scenes(scenes):
    try:
        return [animate_scene_transition(scene1, scene2) for scene1, scene2 in zip(scenes, scenes[1:])]
    except Exception as e:
        st.error(f"Error adding transition effects: {e}")
        return scenes  # Return original scenes if adding transitions fails

# 41. Function to optimize storyboard scene text prompts
def optimize_storyboard_text_prompts(scenes):
    try:
        for scene in scenes:
            scene['title'] = scene['title'].capitalize()
        return scenes
    except Exception as e:
        st.error(f"Error optimizing storyboard text prompts: {e}")
        return scenes  # Return original scenes if optimization fails

# Update the select_background_music function
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
    
    return None  # Return None if no music could be selected

def analyze_script(script):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analyze the script and provide a JSON response with keys: 'sentiment' (POSITIVE, NEGATIVE, or NEUTRAL), 'style' (e.g., formal, casual, humorous), and 'transitions' (list of transition types)."},
                {"role": "user", "content": f"Analyze this script: {script}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        analysis = json.loads(response.choices[0].message.content)
        
        if not all(key in analysis for key in ['sentiment', 'style', 'transitions']):
            raise ValueError("Invalid analysis structure")
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing script: {str(e)}")
        return {
            'sentiment': 'NEUTRAL',
            'style': 'formal',
            'transitions': ['fade', 'cut']
        }

def apply_transition(clip1, clip2, transition_type):
    transition_functions = {
        'fade': lambda: clip1.crossfadeout(1).crossfadein(1),
        'slide': lambda: clip1.slide_out(1, 'left').slide_in(1, 'right'),
        'whip': lambda: clip1.fx(vfx.speedx, 2).fx(vfx.crop, x1=0, y1=0, x2=0.5, y2=1).crossfadeout(0.5),
        'zoom': lambda: clip1.fx(vfx.resize, 1.5).fx(vfx.crop, x_center=0.5, y_center=0.5, width=1/1.5, height=1/1.5).crossfadeout(1)
    }
    return transition_functions.get(transition_type, lambda: clip1)()

def cleanup_temp_files():
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith('videocreator_'):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        logger.info("Temporary files cleaned up successfully.")
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")

def process_scene_with_progress(scene, index, total_scenes):
    scene_progress = st.empty()
    scene_progress.text(f"Processing scene {index + 1} of {total_scenes}: {scene['title']}")
    
    clip_progress, voice_progress = st.columns(2)
    
    with clip_progress:
        st.text("Creating video clip...")
        clip = create_fallback_clip(scene)
        st.success("Video clip processed")
    
    with voice_progress:
        st.text("Generating voiceover...")
        narration_file = generate_voiceover(scene['narration'])
        st.success("Voiceover generated")
    
    scene_progress.success(f"Scene {index + 1} processed successfully!")
    return {'clip': clip, 'scene': scene, 'narration': narration_file}

def generate_valid_storyboard(prompt, style, max_attempts=3):
    for attempt in range(max_attempts):
        storyboard = generate_storyboard(prompt, style)
        if storyboard is not None:
            return storyboard
        logger.warning(f"Storyboard generation attempt {attempt + 1} failed. Retrying...")
    logger.error("Failed to generate a valid storyboard after multiple attempts.")
    st.error("Failed to generate a valid storyboard after multiple attempts. Please try again with a different prompt or style.")
    return None

def prompt_card(prompt):
    st.markdown(f"**Sample Prompt:** {prompt}")
    if st.button("Use this prompt", key=f"btn_{prompt}"):
        st.session_state.prompt = prompt

def predict_processing_issues(video_clips, system_resources):
    potential_issues = []
    if len(video_clips) * 5 > system_resources['available_memory'] / 1e6:  # Assuming 5 seconds per clip
        potential_issues.append("Insufficient memory for processing all clips")
    if system_resources['cpu_usage'] > 80:
        potential_issues.append("High CPU usage may slow down processing")
    return potential_issues

def generate_script(prompt, duration):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional video scriptwriter."},
                {"role": "user", "content": f"Create a storyboard for a {duration}-second video about: {prompt}. Include a title and 5-8 scenes with descriptions."}
            ],
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        return None

def create_video_workflow(prompt, duration, music_style):
    try:
        storyboard = st.session_state.storyboard
        scenes = storyboard['scenes']
        
        video_clips = fetch_video_clips_optimized(scenes)
        
        final_clips = []
        for clip_data in video_clips:
            clip = clip_data['clip']
            scene = clip_data['scene']
            
            # Generate voiceover
            voiceover = generate_voiceover(scene['narration'])
            clip = clip.set_audio(voiceover)
            
            # Add text overlay
            text_clip = mpe.TextClip(scene['title'], fontsize=30, color='white', font='Arial-Bold', size=clip.size)
            text_clip = text_clip.set_position(('center', 'bottom')).set_duration(clip.duration)
            clip = mpe.CompositeVideoClip([clip, text_clip])
            
            final_clips.append(clip)
        
        final_clip = mpe.concatenate_videoclips(final_clips)
        
        # Add background music
        background_music = select_background_music(music_style)
        if background_music:
            background_audio = mpe.AudioFileClip(background_music).volumex(0.1).set_duration(final_clip.duration)
            final_audio = mpe.CompositeAudioClip([final_clip.audio, background_audio])
            final_clip = final_clip.set_audio(final_audio)
        
        # Add intro and outro
        final_clip = add_intro_outro(final_clip, storyboard['title'])
        
        # Write final video file
        output_path = "output_video.mp4"
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=30, preset='faster')
        
        st.success("✅ Video created successfully!")
        st.video(output_path)
        
    except Exception as e:
        st.error(f"An error occurred during video creation: {str(e)}")
    finally:
        cleanup_temp_files()

def main():
    st.markdown("<h1 style='text-align: center; color: #4A90E2;'>AutovideoAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em;'>Create Amazing Videos with AI</p>", unsafe_allow_html=True)

    with st.expander("ℹ️ How to use AutovideoAI", expanded=False):
        st.markdown("""
        1. Enter your video idea or choose a sample prompt.
        2. Customize your video style, duration, and background music.
        3. Generate a storyboard and preview it.
        4. Create your AI-powered video!
        """)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1️⃣ Enter Your Video Idea")
        prompt = st.text_area("What's your video about?", height=100, value=st.session_state.get('prompt', ''))
    
    with col2:
        st.subheader("Sample Prompts")
        for sample_prompt in SAMPLE_PROMPTS:
            if st.button(f"📌 {sample_prompt}", key=f"btn_{sample_prompt}"):
                st.session_state.prompt = sample_prompt
                st.rerun()

    st.subheader("2️⃣ Customize Your Video")
    col1, col2, col3 = st.columns(3)
    with col1:
        style = st.selectbox("Video Style 🎭", ["Motivational", "Dramatic", "Educational", "Funny"])
    with col2:
        duration = st.slider("Estimated Duration ⏱️", 30, 300, 60, help="Duration in seconds")
    with col3:
        music_style = st.selectbox("Background Music 🎵", list(MUSIC_TRACKS.keys()))

    if st.button("🖋️ Generate Storyboard", use_container_width=True):
        with st.spinner("Crafting your storyboard..."):
            storyboard = generate_script(prompt, duration)
        if storyboard:
            st.session_state.storyboard = storyboard
            st.success("✅ Storyboard generated successfully!")
            display_storyboard_preview(storyboard)
        else:
            st.error("❌ Failed to generate a storyboard. Please try again.")

    if 'storyboard' in st.session_state:
        if st.button("🎬 Create Video", use_container_width=True):
            create_video_workflow(prompt, duration, music_style)

def display_storyboard_preview(storyboard):
    with st.expander("🔍 Preview Storyboard", expanded=True):
        st.markdown(f"### {storyboard['title']}")
        st.markdown("### Scene Breakdown")
        for scene in storyboard['scenes']:
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Scene {scene['scene_number']}**")
                    st.write(f"Duration: {scene['duration']} seconds")
                with col2:
                    st.markdown(f"**{scene['title']}**")
                    st.write(f"{scene['description']}")
                st.markdown("---")

# Add this function definition near the top of the file, after the imports
def color_gradient(size, p1, p2, color1, color2):
    x = np.linspace(0, 1, size[0])[:, None]
    y = np.linspace(0, 1, size[1])[None, :]
    gradient = x * (p2[0] - p1[0]) + y * (p2[1] - p1[1])
    gradient = np.clip(gradient, 0, 1)
    return np.array(color1) * (1 - gradient[:, :, None]) + np.array(color2) * gradient[:, :, None]

def download_video(row, output_dir):
    try:
        response = requests.get(row['video_url'], stream=True)
        if response.status_code == 200:
            filename = f"{row['videoid']}.mp4"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return True
    except Exception as e:
        print(f"Error downloading {row['videoid']}: {str(e)}")
    return False

def process_video(filepath, fps=1):
    cap = cv2.VideoCapture(filepath)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def download_videos(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(lambda row: download_video(row, args.output_dir), df.itertuples(index=False)), total=len(df)))

    print(f"Successfully downloaded {sum(results)} videos out of {len(df)}")

    if args.process:
        for filename in os.listdir(args.output_dir):
            if filename.endswith('.mp4'):
                filepath = os.path.join(args.output_dir, filename)
                frames = process_video(filepath, args.fps)
                # Here you can save frames or perform further processing

if __name__ == "__main__":
    main()