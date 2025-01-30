import os
import logging
import numpy as np
import moviepy.editor as mpe
import soundfile as sf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Test directories
TEMP_DIR = "temp_videos"
OUTPUT_DIR = "output_videos"
MUSIC_DIR = "placeholder_music"

# Create directories
for dir_path in [TEMP_DIR, OUTPUT_DIR, MUSIC_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def create_test_video(duration, filename):
    """Creates a test video clip"""
    clip = mpe.ColorClip(size=(720, 480), color=(0, 0, 0), duration=duration)
    clip.write_videofile(filename, fps=24, codec="libx264", audio=False)
    return filename

def create_test_audio():
    """Creates test background music"""
    audio_path = os.path.join(MUSIC_DIR, "background.wav")
    sample_rate = 44100
    duration = 10
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2*np.pi*440*t) * 0.3
    sf.write(audio_path, tone, sample_rate)
    return audio_path

def test_video_assembly():
    try:
        # Create test videos
        video_paths = [
            create_test_video(3, os.path.join(TEMP_DIR, f"test_{i}.mp4"))
            for i in range(2)
        ]
        
        # Create test audio
        music_path = create_test_audio()
        
        # Create simple voiceover
        voiceover_path = os.path.join(TEMP_DIR, "voiceover.wav")
        sample_rate = 44100
        duration = 6
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2*np.pi*880*t) * 0.5
        sf.write(voiceover_path, tone, sample_rate)
        
        # Load clips
        video_clips = [mpe.VideoFileClip(path) for path in video_paths]
        voiceover = mpe.AudioFileClip(voiceover_path)
        background_music = mpe.AudioFileClip(music_path)
        
        # Concatenate videos
        final_video = mpe.concatenate_videoclips(video_clips)
        
        # Mix audio
        final_audio = mpe.CompositeAudioClip([
            voiceover,
            background_music.volumex(0.3)
        ])
        
        # Combine video and audio
        final_video = final_video.set_audio(final_audio)
        
        # Write output
        output_path = os.path.join(OUTPUT_DIR, "test_output.mp4")
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=24
        )
        
        logging.info("Test completed successfully!")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
    finally:
        # Cleanup
        try:
            for clip in video_clips:
                clip.close()
            final_video.close()
            voiceover.close()
            background_music.close()
        except:
            pass

if __name__ == "__main__":
    test_video_assembly() 