import streamlit as st
from openai import OpenAI
import os, requests, tempfile, shutil, json, logging, random, threading, numpy as np, psutil
import moviepy.editor as mpe
import moviepy.video.fx.all as vfx
from PIL import Image, ImageDraw
from gtts import gTTS
from dotenv import load_dotenv
from pydub import AudioSegment, silence
from functools import lru_cache
from huggingface_hub import InferenceClient, hf_hub_download, login, HfApi
from tenacity import retry, wait_random_exponential, stop_after_attempt
from scenedetect import detect, ContentDetector

# 1. Toggle theme
def toggle_theme(): st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# 2. Display prompt card
def prompt_card(prompt: str): st.markdown(f"**Sample Prompt:** {prompt}")

# 3. Main app function
def main():
    st.markdown("<h1 style='text-align: center;'>AutovideoAI</h1>", unsafe_allow_html=True)
    prompt = st.text_area("Enter your video idea:", height=100, key='prompt')
    if st.button("Generate Video") and prompt:
        create_video_workflow(prompt, 60, "Electronic")
    elif not prompt:
        st.error("Please enter a prompt before generating a video.")

# 4. Display storyboard preview
def display_storyboard_preview(storyboard: dict):
    with st.expander("🔎 Preview Storyboard"): st.write(storyboard)

# 5. Create video workflow
def create_video_workflow(prompt: str, duration: int, music_style: str):
    script = generate_script(prompt, duration)
    if script and 'scenes' in script:
        video_clips = [{'clip': create_fallback_clip(scene), 'scene': scene, 'narration': generate_voiceover(scene['narration'])} for scene in script['scenes']]
        background_music = select_background_music(music_style)
        video_file = create_video(video_clips, background_music, script['title']) if background_music else None
        st.video(video_file) if video_file else st.error("Failed to create video. Please try again.")

# 6. Process scene with progress
def process_scene_with_progress(scene: dict, index: int, total_scenes: int):
    return {'clip': create_fallback_clip(scene), 'scene': scene, 'narration': generate_voiceover(scene['narration'])}

# 7. Generate storyboard
def generate_storyboard(prompt: str, style="motivational"):
    response = client.chat_completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"Generate a detailed {style} video storyboard based on this prompt: \"{prompt}\""}], response_format="json", temperature=0.7)
    return json.loads(response.choices[0].message.content) if response else None

# 8. Generate valid storyboard with retries
def generate_valid_storyboard(prompt: str, style: str, max_attempts=3):
    return next((storyboard for _ in range(max_attempts) if (storyboard := generate_storyboard(prompt, style))), None)

# 9. Validate storyboard structure
def validate_storyboard(storyboard: dict):
    return all(key in storyboard for key in ["title", "scenes"]) and isinstance(storyboard["scenes"], list)

# 10. Parse JSON storyboard data
def parse_storyboard(storyboard: str):
    return json.loads(storyboard).get("scenes", []) if storyboard else []

# 11. Save and load storyboard backup
def save_storyboard_backup(storyboard: dict, filename="storyboard_backup.json"):
    if storyboard: json.dump(storyboard, open(filename, 'w'))

def load_storyboard_backup(filename="storyboard_backup.json"):
    return json.load(open(filename)) if os.path.exists(filename) else None

# 12. Split storyboard scenes
def split_storyboard_scenes(scenes: list, batch_size=5):
    return [scenes[i:i + batch_size] for i in range(0, len(scenes), batch_size)]

# 13. Optimize storyboard text prompts
def optimize_storyboard_text_prompts(scenes: list):
    return [{**scene, 'title': scene['title'].capitalize()} for scene in scenes]

# 14. Fetch video clips
def fetch_video_clips(scenes: list):
    return [{'clip': create_fallback_clip(scene), 'scene': scene} for scene in scenes]

# 15. Create scene clip and fallback clip
def create_fallback_clip(scene: dict, duration=5):
    img = Image.new('RGB', (1280, 720), color='black')
    ImageDraw.Draw(img).text((640, 360), scene.get('title', 'Scene'), fill='white', anchor="mm")
    return mpe.ImageClip(np.array(img)).set_duration(duration)

# 16. Enhance and process clip
def enhance_clip(clip): return clip.crossfadein(0.5).crossfadeout(0.5)
def process_clip(enhanced_clip, clip_data, script_analysis):
    return enhanced_clip.set_duration(float(clip_data['scene']['duration']))

# 17. Adjust resolution based on system capacity
def adjust_resolution_based_on_system(clip):
    return clip.resize(newsize=(640, 360) if psutil.virtual_memory().available < 1000 * 1024 * 1024 else (1280, 720))

# 18. Add intro and outro
def add_intro_outro(final_clip, video_title):
    intro = create_animated_text(video_title, duration=3, font_size=60).fadeout(1)
    outro = create_animated_text("Thanks for Watching!", duration=3, font_size=60).fadein(1)
    return mpe.concatenate_videoclips([intro, final_clip, outro])

# 19. Create final video
def create_video(video_clips, background_music_file, video_title):
    final_clip = mpe.concatenate_videoclips([clip['clip'] for clip in video_clips])
    if background_music_file:
        final_clip = final_clip.set_audio(mpe.CompositeAudioClip([final_clip.audio, mpe.AudioFileClip(background_music_file).volumex(0.1)]))
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    return output_file if add_intro_outro(final_clip, video_title).write_videofile(output_file, codec='libx264', audio_codec='aac', fps=24) else None

# 20. Generate voiceover
def generate_voiceover(narration_text: str):
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    gTTS(text=narration_text, lang='en', slow=False).save(temp_audio_file.name)
    return mpe.AudioFileClip(temp_audio_file.name)

# 21. Create silent audio
def create_silent_audio(duration: int):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    AudioSegment.silent(duration=int(duration * 1000)).export(temp_file.name, format="mp3")
    return mpe.AudioFileClip(temp_file.name)

# 22. Add narration and background music
def add_narration(clip, narration_file): return clip.set_audio(mpe.AudioFileClip(narration_file))
def add_background_music(clip, music_file):
    return clip.set_audio(mpe.CompositeAudioClip([clip.audio, mpe.AudioFileClip(music_file).volumex(0.1)]))

# 23. Adjust audio volume
def adjust_audio_volume(audio_clip, volume_level=1.0): return audio_clip.volumex(volume_level)

# 24. Create animated text
def create_animated_text(text: str, duration=5, font_size=70, color='white'):
    img = Image.new('RGB', (1280, 720), color='black')
    ImageDraw.Draw(img).text((640, 360), text, fill=color, anchor="mm")
    return mpe.ImageClip(np.array(img)).set_duration(duration)

# 25. Add fade and other transitions
def add_fade_transition(clip1, clip2, duration=1):
    return mpe.concatenate_videoclips([clip1.crossfadeout(duration), clip2.crossfadein(duration)])

# 26. Apply color grading
def apply_color_grading(clip, brightness=1.0, contrast=1.0, saturation=1.0):
    return clip.fx(vfx.colorx, brightness).fx(vfx.lum_contrast, contrast=contrast).fx(vfx.colorx, saturation)

# 27. Add overlays, lower thirds, watermark, and logo
def add_text_overlay(clip, text):
    txt_clip = mpe.TextClip(text, fontsize=70, color='white').set_duration(clip.duration)
    return mpe.CompositeVideoClip([clip, txt_clip])

def create_lower_third(text, duration):
    img = Image.new('RGBA', (1280, 720), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle(((0, 660), (1280, 720)), fill=(0, 0, 0, 128))
    draw.text((10, 670), text, fill='white')
    return mpe.ImageClip(np.array(img)).set_duration(duration)

def add_watermark(clip, watermark_text="Sample Watermark"):
    watermark = mpe.TextClip(watermark_text, fontsize=30, color='white').set_position(('right', 'bottom')).set_duration(clip.duration)
    return mpe.CompositeVideoClip([clip, watermark])

def add_logo_to_video(clip, logo_path, position=('right', 'top')):
    logo = mpe.ImageClip(logo_path).set_duration(clip.duration).resize(height=100).set_position(position)
    return mpe.CompositeVideoClip([clip, logo])

# 28. Apply filters and overlay image
def apply_bw_filter(clip): return clip.fx(vfx.blackwhite)
def overlay_image_on_video(clip, image_path, position=(0, 0)):
    image = mpe.ImageClip(image_path).set_duration(clip.duration).set_position(position)
    return mpe.CompositeVideoClip([clip, image])

# 29. Split video and merge parts
def split_video(video_clip, part_duration=10):
    return [video_clip.subclip(start, min(start + part_duration, video_clip.duration)) for start in range(0, int(video_clip.duration), part_duration)]

def merge_video_parts(video_parts): return mpe.concatenate_videoclips(video_parts, method="compose")

# 30. Smart cut based on audio silences
def smart_cut(video_clip, audio_clip):
    audio_segment = AudioSegment.from_file(audio_clip) if isinstance(audio_clip, str) else audio_clip.to_soundarray()
    cut_times = [0] + [cut_times[-1] + len(chunk) / 1000 for chunk in silence.split_on_silence(audio_segment, min_silence_len=500, silence_thresh=-40)]
    return mpe.concatenate_videoclips([video_clip.subclip(start, end) for start, end in zip(cut_times[:-1], cut_times[1:])])

# 31. Apply speed changes and crop video
def apply_speed_changes(clip, speed_factor=1.5): return clip.fx(vfx.speedx, speed_factor)
def crop_video(clip, x1, y1, x2, y2): return clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)

# 32. Add subtitles
def add_subtitles_to_video(clip, subtitles):
    return mpe.CompositeVideoClip([clip] + [mpe.TextClip(sub['text'], fontsize=50, color='white').set_position(('bottom')).set_start(sub['start']).set_duration(sub['duration']) for sub in subtitles])

# 33. Download and select background music
def download_additional_assets(url):
    response = requests.get(url)
    return tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name if response.status_code == 200 and response.content else None

def select_background_music(genre):
    MUSIC_TRACKS = {"motivational": "url1", "electronic": "url2"}  # Define this dictionary
    return download_additional_assets(random.choice(list(MUSIC_TRACKS.values())) if genre not in MUSIC_TRACKS else MUSIC_TRACKS[genre])

# 34. Manage temporary files
def cleanup_temp_files():
    [os.unlink(os.path.join(tempfile.gettempdir(), filename)) for filename in os.listdir(tempfile.gettempdir()) if filename.startswith('videocreator_')]

def manage_temp_directory(directory_path):
    if os.path.exists(directory_path): shutil.rmtree(directory_path)
    os.makedirs(directory_path)

# 35. System resource management
def check_system_capabilities():
    if psutil.virtual_memory().available < 500 * 1024 * 1024: st.warning("Low memory detected.")
    if psutil.cpu_percent() > 80: st.warning("High CPU usage detected.")

# 36. Log system resources
def log_system_resources():
    st.write(f"Memory Usage: {psutil.virtual_memory().percent}% | CPU Usage: {psutil.cpu_percent()}%")

# 37. Calculate estimated render time
def calculate_estimated_render_time(duration: int, resolution=(1280, 720)):
    st.info(f"Estimated rendering time: {duration * resolution[0] * resolution[1] / 1e6:.2f} seconds")

# 38. Analyze script
def analyze_script(script: str):
    client = OpenAI()  # Initialize OpenAI client
    response = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": f"Analyze script: {script}"}])
    return json.loads(response.choices[0].message.content) if response else {}

# 39. Predict processing issues
def predict_processing_issues(video_clips: list, system_resources: dict):
    return ["Insufficient memory" if len(video_clips) * 5 > system_resources['available_memory'] / 1e6 else "High CPU usage"]

# 40. Run video rendering thread
def run_video_rendering_thread(target_function, *args):
    threading.Thread(target=target_function, args=args).start()

# 41. Generate video thumbnail
def generate_video_thumbnail(clip, output_path="thumbnail.png"):
    frame = clip.get_frame(1)
    Image.fromarray(frame).save(output_path)
    return output_path

# 42. Compress video
def compress_video(input_path: str, output_path="compressed_video.mp4", bitrate="500k"):
    os.system(f"ffmpeg -i {input_path} -b:v {bitrate} -bufsize {bitrate} {output_path}")

# 43. Generate video script
def generate_script(prompt: str, duration: int):
    client = OpenAI()  # Initialize OpenAI client
    response = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": f"Create script for {duration}s video: {prompt}"}], max_tokens=1000)
    return json.loads(response.choices[0].message.content) if response else {}

# 44. Compress video (duplicate, removed)

# 45. Handle session expiration
def handle_session_expiration(): st.warning("Session expired. Please refresh the page.")

# 46. Log script analysis
def log_script_analysis(analysis_result: dict): st.write("Script Analysis Results:", analysis_result)

# 47. Create lower third (duplicate, removed)

# 48. Generate custom gradient
def generate_gradient(start_color: tuple, end_color: tuple, size: int):
    gradient = Image.new('RGBA', (size, 1), color=start_color)
    ImageDraw.Draw(gradient).rectangle((0, 0, size, 1), fill=end_color)
    return gradient

# 49. Select background color
def select_background_color(scene_type: str):
    colors = {"happy": "yellow", "sad": "blue"}
    return Image.new('RGB', (1280, 720), color=colors.get(scene_type, 'black'))

# 50. Retry failed request
@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def retry_failed_request(func, *args): return func(*args)

# 51. Log video rendering
def log_video_rendering(status: str): st.write(f"Video Rendering Status: {status}")

# 52. Fetch script templates
def fetch_script_templates(category: str):
    templates = {"motivational": "Let's get started!", "intro": "Hello everyone!"}
    return templates.get(category, "Default Template")

# 53. Display generation time
def display_generation_time(duration: float): st.write(f"Generation took {duration:.2f} seconds.")

# 54. Generate ending credits
def generate_ending_credits(credits: str): return create_animated_text(credits, duration=5)

# 55. Merge narration and background
def merge_narration_and_background(clip, narration, background):
    return clip.set_audio(mpe.CompositeAudioClip([narration, background])) if narration and background else clip

# 56. Generate storyboard from prompt
def generate_storyboard_from_prompt(prompt: str):
    storyboard = generate_valid_storyboard(prompt, style="motivational")
    if storyboard and validate_storyboard(storyboard):
        save_storyboard_backup(storyboard)
        return storyboard
    else:
        st.error("Unable to generate storyboard.")
        return None

# 57. Render and display video
def render_and_display_video(storyboard: dict):
    if storyboard:
        display_storyboard_preview(storyboard)
        video_clips = fetch_video_clips(storyboard['scenes'])
        [clip.update({'narration': generate_voiceover(clip['scene']['narration'])}) for clip in video_clips]
        background_music = select_background_music("Electronic")
        video_file = create_video(video_clips, background_music, storyboard['title'])
        if video_file:
            st.video(video_file)
        else:
            st.error("Failed to create video.")
    else:
        st.error("No valid storyboard available to render video.")

# 58. Streamlit interface logic
def streamlit_interface():
    st.markdown("<h1 style='text-align: center;'>AutovideoAI - Generate Stunning Videos</h1>", unsafe_allow_html=True)
    user_prompt = st.text_area("Enter a creative prompt for your video:", key='prompt', height=150)
    if st.button("Generate Storyboard"):
        storyboard = generate_storyboard_from_prompt(user_prompt)
        if storyboard:
            render_and_display_video(storyboard)

# 59. Log and check system health
def log_and_check_system_health():
    log_system_resources()
    check_system_capabilities()

# 60. Handle expired session
def handle_session_timeout():
    if st.session_state.get('expired', False):
        handle_session_expiration()

# Main entry point
if __name__ == "__main__":
    load_dotenv()
    streamlit_interface()
    log_and_check_system_health()
    handle_session_timeout()