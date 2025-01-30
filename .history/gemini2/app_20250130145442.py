import os
import uuid
import asyncio
import logging
import json
from typing import List
import random
import requests
import numpy as np
import moviepy.editor as mpe
import gradio as gr
import yt_dlp
import pysrt
from pydub import AudioSegment
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Configuration ---
VIDEO_OUTPUT_DIR = "output_videos"
TEMP_DIR = "temp_videos"
for dir in [VIDEO_OUTPUT_DIR, TEMP_DIR]:
    if not os.path.exists(dir):
        os.makedirs(dir)

BACKGROUND_MUSIC_PATH = "placeholder_music/background.wav"
if not os.path.exists("placeholder_music"):
    os.makedirs("placeholder_music")
    
# Create a simple sine wave as background music if it doesn't exist
if not os.path.exists(BACKGROUND_MUSIC_PATH):
    sample_rate = 44100
    duration = 60  # 60 seconds of background music
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate a pleasant chord
    tone = np.sin(2*np.pi*440*t) * 0.3 + np.sin(2*np.pi*550*t) * 0.2  # A440 + C#5
    tone = tone * 0.5  # Reduce volume
    sf.write(BACKGROUND_MUSIC_PATH, tone, sample_rate)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_youtube_clips(search_keywords: list, num_clips: int = 3) -> List[str]:
    """Downloads videos using AI-generated keywords."""
    try:
        logging.info(f"Downloading video clips from YouTube for search keywords: {search_keywords[:3]}")
        video_paths = []

        ydl_opts = {
            'format': 'best[ext=mp4][filesize<10M]',  # Reduced from 50MB
            'outtmpl': os.path.join(TEMP_DIR, '%(title)s.%(ext)s'),
            'playlistend': num_clips,  # Correct parameter name
            'quiet': True,
            'match_filter': lambda info: info['duration'] < 30,  # Reduced from 180s
            'max_filesize': 10 * 1024 * 1024  # 10MB
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                for keyword in search_keywords[:3]:  # Use top 3 keywords
                    # Modified YouTube search with keywords
                    results = ydl.extract_info(f"ytsearch{num_clips}:{keyword}", download=True)
                    if 'entries' in results:
                        for entry in results['entries']:
                            if entry:
                                video_path = ydl.prepare_filename(entry)
                                if os.path.exists(video_path) and os.path.getsize(video_path) < 10 * 1024 * 1024:  # 10MB
                                    video_paths.append(video_path)
            except Exception as e:
                logging.error(f"Error downloading from YouTube: {e}")
                return create_placeholder_videos(num_clips)
        
        if not video_paths:
            return create_placeholder_videos(num_clips)
            
        return video_paths
    except Exception as e:
        logging.error(f"Error during video download: {e}")
        return create_placeholder_videos(num_clips)

def create_placeholder_videos(num_clips: int) -> List[str]:
    """Creates multiple placeholder video files."""
    placeholder_files = []
    for i in range(num_clips):
        filename = os.path.join(TEMP_DIR, f'placeholder_{i}.mp4')
        duration = random.randint(3, 5)
        place_holder_filename = create_placeholder_video(duration, filename)
        if place_holder_filename:
            placeholder_files.append(place_holder_filename)
    return placeholder_files

def create_placeholder_video(duration: int, filename: str) -> str:
    """Creates a placeholder video file."""
    try:
        logging.info(f"Creating placeholder video for {duration} seconds")
        # Create a black video
        width, height = 1280, 720
        fps = 24
        
        # Create a black frame
        black_frame = mpe.ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
        
        # Write video file
        black_frame.write_videofile(filename, codec="libx264", audio=False, fps=fps, preset='ultrafast')
        black_frame.close()
        return filename
    except Exception as e:
        logging.error(f"Error creating placeholder video: {e}")
        return None

async def generate_script_from_topic(topic: str) -> dict:
    """Generates structured JSON script with video search keywords."""
    try:
        logging.info(f"Generating AI-powered script for: '{topic}'")
        
        # Example API call - replace with your actual AI service integration
        prompt = f"""Create a 60-second video script about {topic} with:
- 1 title
- 3-5 scenes with visual descriptions 
- 5 search keywords for video clips
- Concise voiceover text for each scene
Output MUST be valid JSON matching this schema:
{{
  "title": str,
  "scenes": [
    {{
      "scene_number": int,
      "visual_description": str,
      "search_keywords": list[str],
      "voiceover_text": str,
      "duration_seconds": float  
    }}
  ],
  "total_duration": float
}}"""
        
        # Example API response simulation
        script_json = {
            "title": f"Exploring {topic}",
            "scenes": [
                {
                    "scene_number": 1,
                    "visual_description": "Opening scene showing key concept",
                    "search_keywords": [f"introduction {topic}", "basic concepts"],
                    "voiceover_text": "Let's begin our journey into...",
                    "duration_seconds": 10.5
                },
                # ... more scenes
            ],
            "total_duration": 60.0
        }
        
        # Validate JSON structure
        if not all(key in script_json for key in ("title", "scenes", "total_duration")):
            raise ValueError("Invalid script structure from AI")
            
        return script_json
        
    except Exception as e:
        logging.error(f"Script generation failed: {e}")
        raise Exception(f"AI script creation error: {e}")

async def generate_voiceover_from_text(text: str) -> str:
    """Generates a simple beep sound as voiceover."""
    try:
        logging.info(f"Generating voiceover audio")
        duration = len(text.split()) * 0.3  # Rough estimate of duration
        sample_rate = 44100
        
        # Generate a simple beep sound
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2*np.pi*440*t)
        
        voiceover_audio_path = os.path.join(TEMP_DIR, "voiceover.wav")
        sf.write(voiceover_audio_path, tone, sample_rate)
        
        return voiceover_audio_path
    except Exception as e:
        logging.error(f"Error during voiceover generation: {e}")
        raise Exception(f"Voiceover generation failed: {e}")

async def assemble_video(video_clip_paths: List[str], voiceover_audio_path: str, music_path: str, output_filepath: str):
    """Assembles the video using MoviePy with given clips, voiceover, and music."""
    try:
        logging.info(f"Assembling video with {len(video_clip_paths)} clips.")
        
        if not video_clip_paths:
            # Create a placeholder video if no clips are available
            placeholder_path = create_placeholder_video(10, os.path.join(TEMP_DIR, "placeholder.mp4"))
            if placeholder_path:
                video_clip_paths = [placeholder_path]
            else:
                raise Exception("Failed to create placeholder video")
        
        video_clips = [mpe.VideoFileClip(clip_path).subclip(0, 5)  # Max 5s per clip
                      for clip_path in video_clip_paths]
        voiceover_clip = mpe.AudioFileClip(voiceover_audio_path)
        background_music_clip = mpe.AudioFileClip(music_path).fx(mpe.afx.audio_normalize)

        # Basic Transitions
        transition_duration = 0.4
        clips_with_transitions = []
        for i, clip in enumerate(video_clips):
            if i > 0:
                clips_with_transitions.append(clip.fx(mpe.transitions.crossfadein, transition_duration).fx(mpe.transitions.crossfadeout, transition_duration))
            else:
                clips_with_transitions.append(clip.fx(mpe.afx.fadein, transition_duration).fx(mpe.afx.fadeout, transition_duration))

        final_video_duration = sum(clip.duration for clip in clips_with_transitions)
        background_music_clip = background_music_clip.subclip(0, final_video_duration).volumex(0.3)
        final_audio = mpe.CompositeAudioClip([voiceover_clip, background_music_clip])
        final_video = mpe.concatenate_videoclips(clips_with_transitions)

        final_video = final_video.set_audio(final_audio)
        final_video = final_video.resize(height=720)  # Limit to 720p

        final_video.write_videofile(output_filepath, 
                                  codec="libx264", 
                                  audio_codec="aac", 
                                  fps=24, 
                                  preset='medium', 
                                  bitrate="1000k",  # Reduced from 3000k
                                  threads=4)
        final_video.close() # Explicitly close video clips to release resources
        voiceover_clip.close()
        background_music_clip.close()
        for clip in video_clips:
             clip.close()
        logging.info("Video assembled successfully.")
    except Exception as e:
       logging.error(f"Error during video assembly: {e}")
       raise Exception(f"Video assembly failed: {e}")
    finally:
        # Ensure resources are cleaned up
        for clip in video_clips:
            try:
                clip.close()
                os.remove(clip.filename)
            except Exception as e:
                logging.warning(f"Error cleaning up {clip.filename}: {e}")


async def generate_subtitles(video_path: str, output_filepath:str):
    """Generates and adds subtitles to the video."""
    try:
         logging.info(f"Generating subtitles for video")
         model = faster_whisper.WhisperModel("large-v2")
         segments, info = model.transcribe(video_path, beam_size=5) # Transcribe audio
         subtitles = []
         for segment in segments:
             subtitles.append(pysrt.SubRipItem(start=segment.start, end=segment.end, text=segment.text)) # create subs
         subtitle_path = output_filepath.replace(".mp4", ".srt")
         pysrt.SubRipFile(subtitles).save(subtitle_path, encoding='utf-8') # Save subtitles

         # add subtitles to the video
         video = mpe.VideoFileClip(video_path)
         subtitles_video = mpe.VideoFileClip(video_path).subtitles(subtitle_path)

         final_clip = mpe.CompositeVideoClip([video, subtitles_video])

         final_clip.write_videofile(output_filepath, codec="libx264", audio_codec="aac", fps=24, preset='medium', bitrate="3000k", threads=4) # Write the final clip
         final_clip.close()
         video.close()
         logging.info("Subtitles generated and added to the video.")
    except Exception as e:
          logging.error(f"Error during subtitle generation: {e}")
          raise Exception(f"Subtitle generation failed: {e}")


async def process_video_generation(topic: str, video_gallery: list) -> str:
    """Main function to orchestrate video generation."""
    logging.info(f"Starting video generation for topic: '{topic}'")
    try:
        # Get structured script
        script_data = await generate_script_from_topic(topic)
        
        # Download clips for each scene
        all_clips = []
        for scene in script_data["scenes"]:
            scene_clips = download_youtube_clips(scene["search_keywords"], num_clips=1)
            all_clips.extend(scene_clips)
        
        # Generate voiceover from actual script text
        voiceover_text = " ".join([s["voiceover_text"] for s in script_data["scenes"]])
        voiceover_path = await generate_voiceover_from_text(voiceover_text)

        video_filename = f"autovideo_{uuid.uuid4()}.mp4"
        output_filepath = os.path.join(VIDEO_OUTPUT_DIR, video_filename)

        await assemble_video(all_clips, voiceover_path, BACKGROUND_MUSIC_PATH, output_filepath)
        await generate_subtitles(output_filepath, output_filepath)

        # Clean up temporary files
        os.remove(voiceover_path)
        for clip_path in all_clips:
            try:
              os.remove(clip_path)
            except:
                pass
        
        # Add to gallery
        video_gallery.append(output_filepath)
                
        logging.info("Video generation completed successfully.")
        return output_filepath

    except Exception as e:
        logging.error(f"Error during video generation process: {e}")
        raise Exception(f"Video generation process failed: {e}")

# --- Gradio UI ---
video_gallery_list = [] # Initialize video gallery list

async def generate_video_ui(topic: str) -> str:
    """Generates video from topic and returns path for Gradio."""
    try:
        output_path = await process_video_generation(topic, video_gallery_list)
        logging.info(f"Generated video at: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error in UI function: {e}")
        return f"Error: {e}"

# FastAPI setup
app = FastAPI()

class VideoRequest(BaseModel):
    topic: str

@app.post("/generate")
async def generate_video_endpoint(request: VideoRequest):
    try:
        output_path = await process_video_generation(request.topic, video_gallery_list)
        return {
            "status": "success",
            "message": "Video generated successfully",
            "video_path": output_path
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

# Create Gradio interface
def load_existing_videos():
    """Loads existing videos from the output directory."""
    existing_videos = []
    for filename in os.listdir(VIDEO_OUTPUT_DIR):
        if filename.endswith(".mp4"):
            existing_videos.append(os.path.join(VIDEO_OUTPUT_DIR, filename))
    return existing_videos

video_gallery_list.extend(load_existing_videos()) # Load existing videos

with gr.Blocks() as demo:
    topic_input = gr.Textbox(label="Enter Video Topic")
    video_output = gr.Video(label="Generated Video")
    video_gallery = gr.Gallery(label="Video Gallery", value=video_gallery_list)

    topic_input.submit(generate_video_ui, inputs=topic_input, outputs=video_output).then(
        lambda: video_gallery_list, outputs=video_gallery
    )

# Launch the Gradio app
if __name__ == "__main__":
    # Create required directories
    for dir_path in [VIDEO_OUTPUT_DIR, TEMP_DIR, "placeholder_music"]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
    # Create placeholder music file if it doesn't exist
    if not os.path.exists(BACKGROUND_MUSIC_PATH):
        with open(BACKGROUND_MUSIC_PATH, 'w') as f:
            f.write("Placeholder music file")
            
    # Mount Gradio app to FastAPI
    app = gr.mount_gradio_app(app, demo, path="/")
    
    # Run FastAPI with uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)