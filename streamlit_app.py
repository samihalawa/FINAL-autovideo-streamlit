import streamlit as st
import openai
import os
import moviepy.editor as mpe
import requests
from tempfile import NamedTemporaryFile
from datasets import load_dataset
from TTS.api import TTS
from moviepy.video.fx.all import fadein, fadeout, resize, crossfadein
import psutil
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
from PIL import Image, ImageDraw, ImageFont
import time
import threading
import shutil
from moviepy.video.tools.drawing import color_gradient

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. Function to generate storyboard based on user prompt using structured JSON
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_storyboard(prompt, style="motivational"):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": (
                "You are a creative assistant that generates detailed video storyboards "
                "based on user prompts. Provide key scenes, narration text for voiceover, "
                "and suggestions for titles and text overlays for each scene in JSON format."
            )},
            {"role": "user", "content": f"Prompt: {prompt}\nStyle: {style}"},
        ]
    )
    return response.get("choices", [{}])[0].get("message", {}).get("content", "{}")

# 2. Function to parse structured JSON storyboard data
def parse_storyboard(storyboard):
    return json.loads(storyboard).get("scenes", [])

# 3. Function to fetch video clips dynamically based on scene keywords
def fetch_video_clips(scenes):
    dataset = load_dataset('GCCPHAT/stock-footage', split='train')
    return [
        {'clip': mpe.VideoFileClip(search_and_download_video(dataset, scene.get('keywords', 'nature')).name), 'scene': scene}
        for scene in scenes if (video_file := search_and_download_video(dataset, scene.get('keywords', 'nature')))
    ]

# 4. Function to search and download video clips based on keywords
def search_and_download_video(dataset, query):
    for item in dataset:
        if query.lower() in item['text'].lower():
            video_response = requests.get(item['url'])
            if video_response.status_code == 200:
                temp_video_file = NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video_file.write(video_response.content)
                temp_video_file.seek(0)
                return temp_video_file
    return None

# 5. Function to generate voiceover with reusable TTS model
def generate_voiceover(narration_text, voice_speed=1.0, pitch=1.0):
    tts_model = initialize_tts_model()
    tts = TTS(tts_model)
    temp_audio_file = NamedTemporaryFile(delete=False, suffix='.wav')
    tts.tts_to_file(text=narration_text, file_path=temp_audio_file.name, speed=voice_speed, pitch=pitch)
    return temp_audio_file.name

# 6. Function to initialize TTS model once
def initialize_tts_model():
    if not hasattr(initialize_tts_model, "tts_model"):
        initialize_tts_model.tts_model = TTS.list_models()[0]
    return initialize_tts_model.tts_model

# 7. Function to create and finalize the video
def create_video(video_clips, narration_file):
    final_clip = mpe.concatenate_videoclips([
        apply_fade_effects(add_text_overlay(clip['clip'], clip['scene']['overlay'])) for clip in video_clips
    ], method='compose')
    return add_narration(final_clip, narration_file).write_videofile("final_video.mp4", fps=24, codec='libx264')

# 8. Function to apply fade-in/fade-out effects to video clips
def apply_fade_effects(clip, duration=1):
    return fadein(clip, duration).fx(fadeout, duration)

# 9. Function to add text overlay to video clips
def add_text_overlay(clip, text):
    return mpe.CompositeVideoClip([clip, mpe.TextClip(text, fontsize=70, color='white', font='Arial-Bold').set_position('center').set_duration(clip.duration)]) if text else clip

# 10. Function to add narration to video clip
def add_narration(clip, narration_file):
    return clip.set_audio(mpe.AudioFileClip(narration_file))

# 11. Function to handle errors and notify the user
def handle_error(error_message):
    st.error(f"An error occurred: {error_message}")

# 12. Function to display real-time progress in the UI
def show_progress(stage_message):
    st.info(stage_message)

# 13. Function to clear temporary files after processing
def cleanup_temp_files(*files):
    [os.remove(file) for file in files if os.path.exists(file)]

# 14. Function to show video preview in the app
def preview_video(video_path):
    with open(video_path, 'rb') as video_file:
        st.video(video_file.read())

# 15. Function to apply additional video effects (brightness, contrast, etc.)
def apply_video_effects(clip, brightness=1.0, contrast=1.0):
    return clip.fx(mpe.vfx.colorx, brightness).fx(mpe.vfx.lum_contrast, contrast=contrast)

# 16. Function to add watermarks to video clips
def add_watermark(clip, watermark_text="Sample Watermark"):
    watermark = mpe.TextClip(watermark_text, fontsize=30, color='white', font='Arial')
    watermark = watermark.set_position(('right', 'bottom')).set_duration(clip.duration)
    return mpe.CompositeVideoClip([clip, watermark])

# 17. Function to create animated text sequences
def create_animated_text(text, duration=5):
    txt_clip = mpe.TextClip(text, fontsize=70, color='yellow', font='Arial-Bold', kerning=5)
    txt_clip = txt_clip.set_position('center').set_duration(duration).fadein(1).fadeout(1)
    return txt_clip

# 18. Function to download additional video assets (e.g., background music)
def download_additional_assets(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_asset_file = NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_asset_file.write(response.content)
        temp_asset_file.seek(0)
        return temp_asset_file.name
    return None

# 19. Function to add background music to video
def add_background_music(clip, music_file):
    background_audio = mpe.AudioFileClip(music_file)
    return clip.set_audio(mpe.CompositeAudioClip([clip.audio, background_audio.volumex(0.1)]))

# 20. Function to split video into parts for processing
def split_video(video_clip, part_duration=10):
    return [video_clip.subclip(start, min(start + part_duration, video_clip.duration)) for start in range(0, int(video_clip.duration), part_duration)]

# 21. Function to merge video parts back together
def merge_video_parts(video_parts):
    return mpe.concatenate_videoclips(video_parts, method="compose")

# 22. Function to monitor and log system resources during video generation
def log_system_resources():
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    st.write(f"Memory Usage: {memory.percent}% | CPU Usage: {cpu}%")

# 23. Function to save a temporary JSON backup of generated storyboard
def save_storyboard_backup(storyboard, filename="storyboard_backup.json"):
    with open(filename, 'w') as f:
        json.dump(storyboard, f)

# 24. Function to load a saved storyboard from backup
def load_storyboard_backup(filename="storyboard_backup.json"):
    with open(filename, 'r') as f:
        return json.load(f)

# 25. Function to adjust audio volume levels
def adjust_audio_volume(audio_clip, volume_level=1.0):
    return audio_clip.volumex(volume_level)

# 26. Function to animate scene transitions
def animate_scene_transition(clip1, clip2, duration=1):
    return mpe.concatenate_videoclips([clip1.crossfadeout(duration), clip2.crossfadein(duration)])

# 27. Function to crop video clips
def crop_video(clip, x1, y1, x2, y2):
    return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)

# 28. Function to calculate estimated video rendering time
def calculate_estimated_render_time(duration, resolution=(1280, 720)):
    return duration * (resolution[0] * resolution[1]) / 1e6

# 29. Function to run video rendering in a separate thread
def run_video_rendering_thread(target_function, *args):
    rendering_thread = threading.Thread(target=target_function, args=args)
    rendering_thread.start()
    return rendering_thread

# 30. Function to manage temporary directories
def manage_temp_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

# 31. Function to provide video editing tips in the UI
def show_video_editing_tips():
    st.sidebar.header("Video Editing Tips")
    st.sidebar.text("- Keep it short and engaging.")
    st.sidebar.text("- Use consistent fonts and colors.")
    st.sidebar.text("- Add background music for impact.")

# 32. Function to add intro and outro sequences to video
def add_intro_outro(clip, intro_text="Welcome", outro_text="Thank you for watching!"):
    intro = create_animated_text(intro_text, duration=3)
    outro = create_animated_text(outro_text, duration=3)
    return mpe.concatenate_videoclips([intro, clip, outro])

# 33. Function to adjust video speed
def adjust_video_speed(clip, speed=1.0):
    return clip.fx(mpe.vfx.speedx, speed)

# 34. Function to overlay images on video
def overlay_image_on_video(clip, image_path, position=(0, 0)):
    image = mpe.ImageClip(image_path).set_duration(clip.duration).set_position(position)
    return mpe.CompositeVideoClip([clip, image])

# 35. Function to generate thumbnail for video
def generate_video_thumbnail(clip, output_path="thumbnail.png"):
    frame = clip.get_frame(1)
    image = Image.fromarray(frame)
    image.save(output_path)
    return output_path

# 36. Function to check system capabilities before rendering
def check_system_capabilities():
    memory = psutil.virtual_memory()
    if memory.available < 500 * 1024 * 1024:  # Less than 500MB
        st.warning("Low memory detected. Consider closing other applications.")

# 37. Function to generate a text overlay with gradient background
def generate_gradient_text_overlay(text, clip_duration, size=(1920, 1080)):
    gradient = color_gradient(size, p1=(0, 0), p2=(size[0], size[1]), color1=(255, 0, 0), color2=(0, 0, 255))
    image = Image.fromarray(gradient)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_size = draw.textsize(text, font=font)
    draw.text(((size[0] - text_size[0]) / 2, (size[1] - text_size[1]) / 2), text, font=font, fill=(255, 255, 255))
    image.save("gradient_overlay.png")
    return mpe.ImageClip("gradient_overlay.png").set_duration(clip_duration)

# 38. Function to add subtitles to video
def add_subtitles_to_video(clip, subtitles):
    subtitle_clips = [
        mpe.TextClip(subtitle['text'], fontsize=50, color='white', size=clip.size, font='Arial-Bold')
        .set_position(('bottom')).set_start(subtitle['start']).set_duration(subtitle['duration'])
        for subtitle in subtitles
    ]
    return mpe.CompositeVideoClip([clip] + subtitle_clips)

# 39. Function to preview storyboard as a slideshow
def preview_storyboard_slideshow(scenes, duration_per_scene=5):
    slides = [create_animated_text(scene['title'], duration=duration_per_scene) for scene in scenes]
    slideshow = mpe.concatenate_videoclips(slides, method='compose')
    slideshow.write_videofile("storyboard_preview.mp4", fps=24, codec='libx264')
    preview_video("storyboard_preview.mp4")

# 40. Function to add logo to video
def add_logo_to_video(clip, logo_path, position=('right', 'top')):
    logo = mpe.ImageClip(logo_path).set_duration(clip.duration).resize(height=100).set_position(position)
    return mpe.CompositeVideoClip([clip, logo])

# 41. Function to compress video output for faster uploading
def compress_video(input_path, output_path="compressed_video.mp4", bitrate="500k"):
    os.system(f"ffmpeg -i {input_path} -b:v {bitrate} -bufsize {bitrate} {output_path}")
    return output_path

# 42. Function to adjust resolution dynamically based on system capacity
def adjust_resolution_based_on_system(clip):
    memory = psutil.virtual_memory()
    resolution = (640, 360) if memory.available < 1000 * 1024 * 1024 else (1280, 720)
    return resize(clip, newsize=resolution)

# 43. Function to handle session expiration or token errors
def handle_session_expiration():
    st.error("Session expired. Please refresh and try again.")
    st.button("Refresh Page")

# 44. Function to apply black-and-white filter to video
def apply_bw_filter(clip):
    return clip.fx(mpe.vfx.blackwhite)

# 45. Function to display estimated storage requirements
def display_estimated_storage_requirements(video_length_seconds, resolution=(1280, 720)):
    storage_estimate = (video_length_seconds * resolution[0] * resolution[1] * 0.1) / 1e6  # Estimate in MB
    st.write(f"Estimated Storage Requirement: {storage_estimate:.2f} MB")

# Main function to run the Streamlit app
def main():
    st.title("🎬 AI-Powered Video Editor")
    prompt = st.text_area("Enter your video idea:", height=150)
    style = st.selectbox("Select a storyboard style", ["motivational", "dramatic", "educational", "funny"])
    voice_speed = st.slider("Select voice speed", 0.5, 2.0, 1.0)
    voice_pitch = st.slider("Select voice pitch", 0.5, 2.0, 1.0)

    if st.button("Generate Video"):
        if not prompt.strip():
            st.error("Please enter a valid video idea.")
            return

        try:
            show_progress("Generating storyboard...")
            storyboard = generate_storyboard(prompt, style=style)
            if not storyboard:
                st.error("Failed to generate storyboard.")
                return
            st.success("Storyboard generated.")
            st.subheader("Generated Storyboard")
            st.text(storyboard)

            show_progress("Parsing storyboard...")
            scenes = parse_storyboard(storyboard)
            if not scenes:
                handle_error("Failed to parse storyboard.")
                return
            st.success("Storyboard parsed successfully.")

            narration_text = ' '.join([scene['narration'] for scene in scenes if 'narration' in scene])
            if not narration_text:
                handle_error("No narration text found in storyboard.")
                return

            show_progress("Fetching video clips...")
            video_clips = fetch_video_clips(scenes)
            if not video_clips:
                st.error("No video clips fetched.")
                return
            st.success("Video clips fetched.")

            show_progress("Generating voiceover...")
            narration_file = generate_voiceover(narration_text, voice_speed=voice_speed, pitch=voice_pitch)
            st.success("Voiceover generated.")

            show_progress("Creating final video...")
            create_video(video_clips, narration_file)
            st.success("Video created successfully!")

            preview_video("final_video.mp4")
            st.download_button(label="Download Video", data=open("final_video.mp4", 'rb'), file_name="final_video.mp4")
            cleanup_temp_files(narration_file)

        except Exception as e:
            handle_error(str(e))

if __name__ == "__main__":
    main()
