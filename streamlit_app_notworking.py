import streamlit as st
import os
import logging
import tempfile
from pathlib import Path
import yt_dlp
import moviepy.editor as mpe
from typing import List, Dict
import concurrent.futures
import json
import shutil

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoConfig:
    STYLES = {
        "Motivational": {
            "description": "Inspiring and uplifting content",
            "transitions": ["fade", "dissolve"],
            "keywords": ["inspiration", "success", "achievement"]
        },
        "Educational": {
            "description": "Clear, informative content",
            "transitions": ["cut", "fade"],
            "keywords": ["learning", "explanation", "demonstration"]
        },
        "Cinematic": {
            "description": "Dramatic and visually rich",
            "transitions": ["fade", "crossfade"],
            "keywords": ["dramatic", "scenic", "atmospheric"]
        }
    }

class Scene:
    def __init__(self, description: str, keywords: List[str], duration: float = 10.0):
        self.description = description
        self.keywords = keywords
        self.duration = duration
        self.video_path = None

class VideoGenerator:
    def __init__(self):
        """Initialize with temp directory for video processing"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ydl_opts = {
            'format': 'best[ext=mp4][height<=720]',
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }

    def generate_scenes(self, prompt: str, style: str) -> List[Scene]:
        """Generate scenes from prompt using rule-based approach"""
        scenes = []
        segments = [s.strip() for s in prompt.split('.') if s.strip()]
        style_keywords = VideoConfig.STYLES[style]["keywords"]

        for segment in segments:
            # Extract keywords from segment
            keywords = [word.lower() for word in segment.split() 
                       if len(word) > 3] + style_keywords
            scenes.append(Scene(
                description=segment,
                keywords=list(set(keywords))  # Remove duplicates
            ))
        return scenes

    def fetch_video(self, scene: Scene) -> str:
        """Fetch video from multiple sources"""
        sources = [
            self._fetch_from_youtube,
            self._fetch_from_archive
        ]

        for source_func in sources:
            try:
                video_path = source_func(scene)
                if video_path:
                    return video_path
            except Exception as e:
                logger.error(f"Error in source {source_func.__name__}: {e}")
        
        return None

    def _fetch_from_youtube(self, scene: Scene) -> str:
        """Fetch video from YouTube"""
        search_query = f"Creative Commons {' '.join(scene.keywords)}"
        
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            try:
                result = ydl.extract_info(
                    f"ytsearch1:{search_query}",
                    download=True
                )
                if result and 'entries' in result and result['entries']:
                    return ydl.prepare_filename(result['entries'][0])
            except Exception as e:
                logger.error(f"YouTube download error: {e}")
        return None

    def _fetch_from_archive(self, scene: Scene) -> str:
        """Fetch video from Internet Archive"""
        # Internet Archive implementation
        # Currently a placeholder for future implementation
        return None

    def create_final_video(self, scenes: List[Scene], style: str) -> str:
        """Create final video with transitions"""
        clips = []
        transitions = VideoConfig.STYLES[style]["transitions"]

        for scene in scenes:
            if scene.video_path and os.path.exists(scene.video_path):
                try:
                    clip = mpe.VideoFileClip(scene.video_path)
                    # Trim clip to scene duration
                    clip = clip.subclip(0, min(clip.duration, scene.duration))
                    
                    # Add fade in/out
                    clip = clip.fx(mpe.vfx.fadein, duration=1)
                    clip = clip.fx(mpe.vfx.fadeout, duration=1)
                    
                    clips.append(clip)
                except Exception as e:
                    logger.error(f"Error processing clip {scene.video_path}: {e}")

        if clips:
            try:
                final_video = mpe.concatenate_videoclips(clips, method="compose")
                output_path = str(self.temp_dir / "final_video.mp4")
                final_video.write_videofile(output_path, fps=24)
                return output_path
            except Exception as e:
                logger.error(f"Error creating final video: {e}")

        return None

    def cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")

def main():
    st.set_page_config(
        page_title="Open Source Video Generator",
        layout="wide"
    )

    st.title("🎬 AI Video Generator")
    st.write("Create videos using open-source content")

    # User Input
    with st.form("video_form"):
        prompt = st.text_area(
            "Describe your video",
            placeholder="Enter your video description. Each sentence will become a scene.",
            help="Be specific and descriptive for better results."
        )

        style = st.selectbox(
            "Choose Style",
            options=list(VideoConfig.STYLES.keys()),
            format_func=lambda x: f"{x}: {VideoConfig.STYLES[x]['description']}"
        )

        submitted = st.form_submit_button("Generate Video")

    if submitted and prompt:
        generator = VideoGenerator()
        
        try:
            # Progress tracking
            progress = st.progress(0)
            status = st.empty()

            # Generate scenes
            status.text("Generating scenes...")
            scenes = generator.generate_scenes(prompt, style)

            # Display scene information
            st.subheader("Scenes")
            for i, scene in enumerate(scenes, 1):
                with st.expander(f"Scene {i}"):
                    st.write(f"Description: {scene.description}")
                    st.write(f"Keywords: {', '.join(scene.keywords)}")

            # Fetch videos
            status.text("Fetching videos...")
            for i, scene in enumerate(scenes):
                progress.progress((i + 1) / (len(scenes) * 2))
                scene.video_path = generator.fetch_video(scene)
                if scene.video_path:
                    st.success(f"Found video for scene {i+1}")
                else:
                    st.warning(f"No video found for scene {i+1}")

            # Create final video
            if any(scene.video_path for scene in scenes):
                status.text("Creating final video...")
                progress.progress(0.75)
                
                final_path = generator.create_final_video(scenes, style)
                
                if final_path:
                    progress.progress(1.0)
                    status.text("Video generated successfully!")
                    
                    # Display video
                    st.video(final_path)
                    
                    # Download button
                    with open(final_path, 'rb') as f:
                        st.download_button(
                            "Download Video",
                            f,
                            file_name="generated_video.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("Failed to create final video")
            else:
                st.error("No suitable videos found")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main execution: {e}")

        finally:
            # Cleanup
            generator.cleanup()

if __name__ == "__main__":
    main()