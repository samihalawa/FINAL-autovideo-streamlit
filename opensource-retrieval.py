from dataclasses import dataclass
import streamlit as st
import yt_dlp
import moviepy.editor as mpe
from typing import List, Dict
import logging
import os
from pathlib import Path

@dataclass
class Scene:
    description: str
    keywords: List[str]
    duration: float
    style: str

@dataclass
class VideoClip:
    path: str
    source: str
    duration: float
    scene: Scene

class VideoGenerator:
    def __init__(self):
        self.temp_dir = Path("./temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def generate_scenes(self, prompt: str, style: str) -> List[Scene]:
        """Generate scene descriptions from prompt"""
        # Simple rule-based scene generation
        scenes = []
        # Split prompt into natural segments
        segments = prompt.split('.')
        for segment in segments:
            if segment.strip():
                # Extract keywords from segment
                keywords = [word.strip().lower() for word in segment.split() 
                          if len(word.strip()) > 3]
                scenes.append(Scene(
                    description=segment.strip(),
                    keywords=keywords,
                    duration=10.0,  # Default duration
                    style=style
                ))
        return scenes

    def fetch_videos(self, scene: Scene) -> List[VideoClip]:
        """Fetch videos for a scene using yt-dlp"""
        search_query = f"{scene.description} {scene.style}"
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'max_downloads': 3
        }
        
        clips = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Search YouTube
                results = ydl.extract_info(
                    f"ytsearch3:{search_query}", 
                    download=True
                )
                for entry in results['entries']:
                    if entry:
                        clip = VideoClip(
                            path=ydl.prepare_filename(entry),
                            source="youtube",
                            duration=entry['duration'],
                            scene=scene
                        )
                        clips.append(clip)
            except Exception as e:
                self.logger.error(f"Error fetching videos: {e}")
        
        return clips

    def create_final_video(self, clips: List[VideoClip], 
                          output_path: str) -> str:
        """Combine video clips into final video"""
        final_clips = []
        for clip in clips:
            try:
                video = mpe.VideoFileClip(clip.path)
                # Trim to scene duration
                video = video.subclip(0, min(clip.scene.duration, 
                                           video.duration))
                final_clips.append(video)
            except Exception as e:
                self.logger.error(f"Error processing clip {clip.path}: {e}")
        
        if final_clips:
            final_video = mpe.concatenate_videoclips(final_clips)
            final_video.write_videofile(output_path)
            return output_path
        return None

def main():
    st.set_page_config(page_title="AI Video Generator", 
                      layout="wide")
    
    st.title("🎬 AI Video Generator")
    
    # User Input Section
    with st.form("video_generator_form"):
        prompt = st.text_area(
            "Enter your video concept",
            placeholder="Example: A journey through nature, starting with a peaceful forest, then mountains, and ending at the ocean."
        )
        
        style = st.selectbox(
            "Choose video style",
            ["Cinematic", "Documentary", "Artistic"]
        )
        
        submitted = st.form_submit_button("Generate Video")
    
    if submitted and prompt:
        generator = VideoGenerator()
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate Scenes
        status_text.text("Generating scenes...")
        scenes = generator.generate_scenes(prompt, style)
        
        # Display scenes
        st.subheader("Generated Scenes")
        for i, scene in enumerate(scenes, 1):
            st.write(f"Scene {i}: {scene.description}")
        
        # Fetch videos
        status_text.text("Fetching videos...")
        all_clips = []
        for i, scene in enumerate(scenes):
            progress_bar.progress((i + 1) / len(scenes))
            clips = generator.fetch_videos(scene)
            all_clips.extend(clips)
        
        # Create final video
        if all_clips:
            status_text.text("Creating final video...")
            output_path = "final_video.mp4"
            final_path = generator.create_final_video(all_clips, output_path)
            
            if final_path:
                status_text.text("Video generated successfully!")
                st.video(final_path)
            else:
                st.error("Failed to generate video")
        else:
            st.error("No suitable video clips found")

if __name__ == "__main__":
    main()