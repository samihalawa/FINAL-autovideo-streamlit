# ai_video_editor_final_extended_production_ready.py

import streamlit as st
import openai
import os
import moviepy.editor as mpe
import requests
from tempfile import NamedTemporaryFile
from datasets import load_dataset
from tts import TTS
from moviepy.video.fx.all import fadein, fadeout, resize, crossfadein
import psutil

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. Function to generate storyboard based on user prompt using structured JSON
def generate_storyboard(prompt, style="motivational"):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a creative assistant that generates detailed video storyboards "
                    "based on user prompts. Provide key scenes, narration text for voiceover, "
                    "and suggestions for titles and text overlays for each scene in JSON format."
                ),
            },
            {"role": "user", "content": f"Prompt: {prompt}\nStyle: {style}"},
        ],
        functions=[
            {
                "name": "generate_storyboard",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scenes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "narration": {"type": "string"},
                                    "overlay": {"type": "string"},
                                    "keywords": {"type": "string"},
                                },
                                "required": ["title", "narration", "overlay", "keywords"],
                            },
                        },
                    },
                    "required": ["scenes"],
                },
            }
        ],
    )
    storyboard = response["choices"][0]["message"]["function_call"]["arguments"]
    return storyboard

# 2. Function to parse structured JSON storyboard data
def parse_storyboard(storyboard):
    return storyboard.get("scenes", [])

# 3. Function to fetch video clips dynamically based on scene keywords
def fetch_video_clips(scenes):
    video_clips = []
    dataset = load_dataset('GCCPHAT/stock-footage', split='train')
    
    for idx, scene in enumerate(scenes):
        keywords = scene.get('keywords') or scene.get('overlay') or 'nature'
        st.info(f"Fetching video for Scene {idx+1}: {keywords}")
        video_file = search_and_download_video(dataset, keywords)
        if video_file:
            clip = mpe.VideoFileClip(video_file.name)
            video_clips.append({'clip': clip, 'scene': scene})
        else:
            st.warning(f"No specific video found for: {keywords}. Using a placeholder.")
    return video_clips

# 4. Function to search and download video clips based on keywords
def search_and_download_video(dataset, query):
    for item in dataset:
        if query.lower() in item['text'].lower():
            video_url = item['url']
            video_response = requests.get(video_url)
            if video_response.status_code == 200:
                temp_video_file = NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video_file.write(video_response.content)
                temp_video_file.seek(0)
                return temp_video_file
    return None

# 5. Function to generate voiceover with custom options (speed, pitch)
def generate_voiceover(narration_text, voice_speed=1.0, pitch=1.0):
    tts_model = TTS.list_models()[0]
    tts = TTS(tts_model)
    temp_audio_file = NamedTemporaryFile(delete=False, suffix='.wav')
    tts.tts_to_file(text=narration_text, file_path=temp_audio_file.name, speed=voice_speed, pitch=pitch)
    return temp_audio_file.name

# 6. Function to apply fade-in/fade-out effects to video clips
def apply_fade_effects(clip, duration=1):
    return fadein(clip, duration).fx(fadeout, duration)

# 7. Function to add text overlay to video clips
def add_text_overlay(clip, text):
    if text:
        txt_clip = mpe.TextClip(text, fontsize=70, color='white', font='Arial-Bold')
        txt_clip = txt_clip.set_position('center').set_duration(clip.duration)
        clip = mpe.CompositeVideoClip([clip, txt_clip])
    return clip

# 8. Function to resize video clips if needed
def resize_video(clip, size=(1280, 720)):
    return resize(clip, newsize=size)

# 9. Function to concatenate multiple clips into one video
def concatenate_clips(clips):
    return mpe.concatenate_videoclips(clips, method='compose')

# 10. Function to add crossfade transition between clips
def crossfade_transition(clip1, clip2, duration=1):
    return clip1.crossfadein(duration).fx(crossfadein, clip2, duration)

# 11. Function to synchronize narration audio with video
def add_narration(clip, narration_file):
    narration = mpe.AudioFileClip(narration_file)
    return clip.set_audio(narration)

# 12. Function to handle errors and notify the user
def handle_error(error_message):
    st.error(f"An error occurred: {error_message}")

# 13. Function to create and finalize the video
def create_video(video_clips, narration_file):
    clips = []
    for vc in video_clips:
        clip = vc['clip']
        scene = vc['scene']
        clip = add_text_overlay(clip, scene['overlay'])
        clip = apply_fade_effects(clip)
        clip = resize_video(clip)
        clips.append(clip)

    final_clip = concatenate_clips(clips)
    final_clip = add_narration(final_clip, narration_file)
    final_video_path = "final_video.mp4"
    final_clip.write_videofile(final_video_path, fps=24, codec='libx264')
    return final_video_path

# 14. Function to display real-time progress in the UI
def show_progress(stage_message):
    st.info(stage_message)

# 15. Function to clear temporary files after processing
def cleanup_temp_files(*files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)

# 16. Function to show video preview in the app
def preview_video(video_path):
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)

# 17. Function to fetch user settings (voice pitch, speed)
def fetch_user_settings():
    voice_speed = st.slider("Select voice speed", 0.5, 2.0, 1.0)
    voice_pitch = st.slider("Select voice pitch", 0.5, 2.0, 1.0)
    return voice_speed, voice_pitch

# 18. Function to process storyboard details into structured data
def process_storyboard_details(scenes):
    parsed_scenes = []
    for scene in scenes:
        parsed_scene = {
            'title': scene['title'],
            'narration': scene['narration'],
            'overlay': scene['overlay'],
            'keywords': scene['keywords']
        }
        parsed_scenes.append(parsed_scene)
    return parsed_scenes

# 19. Function to validate the prompt provided by the user
def validate_prompt(prompt):
    if not prompt or len(prompt.strip()) == 0:
        return False
    return True

# 20. Function to display storyboard details in a formatted way
def display_storyboard_details(scenes):
    st.subheader("Generated Storyboard Details")
    for idx, scene in enumerate(scenes):
        st.write(f"**Scene {idx+1}:** {scene['title']}")
        st.write(f"**Narration:** {scene['narration']}")
        st.write(f"**Overlay:** {scene['overlay']}")
        st.write(f"**Keywords:** {scene['keywords']}")

# 21. Function to adjust the output video resolution
def adjust_video_resolution(clip, resolution=(1280, 720)):
    return resize(clip, resolution)

# 22. Function to calculate estimated video length
def calculate_estimated_video_length(clips):
    total_duration = sum([clip.duration for clip in clips])
    return total_duration

# 23. Function to generate additional effects for video clips (color, brightness, etc.)
def apply_additional_video_effects(clip, brightness=1.0, contrast=1.0):
    # Apply brightness and contrast effects to the clip (logic to be added)
    return clip

# 24. Function to ensure TTS models are loaded only once
def initialize_tts_model():
    if not hasattr(initialize_tts_model, "tts_model"):
        initialize_tts_model.tts_model = TTS.list_models()[0]
    return initialize_tts_model.tts_model

# 25. Function to generate voiceover with reusable TTS model
def generate_voiceover_with_reusable_model(narration_text, voice_speed=1.0, pitch=1.0):
    tts_model = initialize_tts_model()
    tts = TTS(tts_model)
    temp_audio_file = NamedTemporaryFile(delete=False, suffix='.wav')
    tts.tts_to_file(text=narration_text, file_path=temp_audio_file.name, speed=voice_speed, pitch=pitch)
    return temp_audio_file.name

# 26. Function to download additional video assets
def download_additional_video_assets(asset_url):
    response = requests.get(asset_url)
    if response.status_code == 200:
        temp_asset_file = NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_asset_file.write(response.content)
        temp_asset_file.seek(0)
        return temp_asset_file
    return None

# 27. Function to check for any missing video or audio assets before final rendering
def check_for_missing_assets(video_clips, narration_file):
    if not video_clips or not narration_file:
        return False
    return True

# 28. Function to create a backup of generated storyboard
def backup_storyboard(storyboard, backup_path="backup_storyboard.json"):
    with open(backup_path, 'w') as backup_file:
        backup_file.write(storyboard)

# 29. Function to allow users to download the final video
def download_video(video_path):
    with open(video_path, 'rb') as video_file:
        st.download_button(label="Download Video", data=video_file, file_name="final_video.mp4")

# 30. Function to track and display progress during video generation
def track_generation_progress(current_stage, total_stages):
    st.progress((current_stage / total_stages) * 100)

# 31. Function to generate a storyboard using custom styles
def generate_custom_style_storyboard(prompt, style="motivational"):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a creative assistant generating detailed video storyboards "
                    "based on user prompts in a given style. The style can be motivational, dramatic, "
                    "educational, or funny. Please return the output in JSON format."
                ),
            },
            {"role": "user", "content": f"Prompt: {prompt}\nStyle: {style}"},
        ],
    )
    storyboard = response["choices"][0]["message"]["function_call"]["arguments"]
    return storyboard

# 32. Function to monitor system resources while generating videos
def monitor_system_resources():
    memory = psutil.virtual_memory()
    st.write(f"Memory Usage: {memory.percent}%")
    if memory.percent > 80:
        st.warning("High memory usage detected. Consider reducing video quality or length.")

# 33. Main function to run the Streamlit app
def main():
    st.title("🎬 AI-Powered Video Editor")
    st.write("Describe your video idea, and watch it come to life!")
    
    prompt = st.text_area("Enter your video idea:", height=150)
    style = st.selectbox("Select a storyboard style", ["motivational", "dramatic", "educational", "funny"])
    voice_speed, voice_pitch = fetch_user_settings()

    if st.button("Generate Video"):
        if not validate_prompt(prompt):
            st.error("Please enter a valid video idea.")
            return

        try:
            show_progress("Generating storyboard...")
            storyboard = generate_custom_style_storyboard(prompt, style=style)
            st.success("Storyboard generated.")
            st.subheader("Generated Storyboard")
            st.text(storyboard)

            show_progress("Parsing storyboard...")
            scenes = parse_storyboard(storyboard)
            if not scenes:
                handle_error("Failed to parse storyboard.")
                return
            st.success("Storyboard parsed successfully.")
            display_storyboard_details(scenes)

            narration_text = ' '.join([scene['narration'] for scene in scenes if scene['narration']])
            if not narration_text:
                handle_error("No narration text found in storyboard.")
                return

            show_progress("Fetching video clips...")
            video_clips = fetch_video_clips(scenes)
            st.success("Video clips fetched.")

            show_progress("Generating voiceover...")
            narration_file = generate_voiceover_with_reusable_model(narration_text, voice_speed=voice_speed, pitch=voice_pitch)
            st.success("Voiceover generated.")

            show_progress("Creating final video...")
            final_video_path = create_video(video_clips, narration_file)
            st.success("Video created successfully!")

            preview_video(final_video_path)
            download_video(final_video_path)
            cleanup_temp_files(narration_file)

        except Exception as e:
            handle_error(str(e))

if __name__ == "__main__":
    main()
