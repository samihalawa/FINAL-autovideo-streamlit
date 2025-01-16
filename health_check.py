import streamlit as st
import os
import tempfile
import shutil
import requests
from openai import OpenAI
from gtts import gTTS
import moviepy.editor as mpe

def check_openai_api():
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return True, "OpenAI API connection successful"
    except Exception as e:
        return False, f"OpenAI API error: {str(e)}"

def check_disk_space(min_space_mb=500):
    try:
        _, _, free = shutil.disk_usage(tempfile.gettempdir())
        free_mb = free // (2**20)
        return free_mb >= min_space_mb, f"Available disk space: {free_mb}MB"
    except Exception as e:
        return False, f"Disk space check error: {str(e)}"

def check_music_endpoints():
    try:
        test_url = "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Tours/Enthusiast/Tours_-_01_-_Enthusiast.mp3"
        response = requests.head(test_url, timeout=5)
        return response.status_code == 200, "Music endpoints accessible"
    except Exception as e:
        return False, f"Music endpoint error: {str(e)}"

def check_text_to_speech():
    try:
        tts = gTTS(text="test", lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        os.unlink(temp_file.name)
        return True, "Text-to-speech working"
    except Exception as e:
        return False, f"Text-to-speech error: {str(e)}"

def check_video_processing():
    try:
        clip = mpe.ColorClip(size=(128, 128), color=(0,0,0), duration=1)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        clip.write_videofile(temp_file.name, fps=24, logger=None)
        os.unlink(temp_file.name)
        return True, "Video processing working"
    except Exception as e:
        return False, f"Video processing error: {str(e)}"

def main():
    st.title("🏥 System Health Check")
    
    checks = {
        "OpenAI API": check_openai_api,
        "Disk Space": check_disk_space,
        "Music Endpoints": check_music_endpoints,
        "Text-to-Speech": check_text_to_speech,
        "Video Processing": check_video_processing
    }
    
    if st.button("Run Health Check", type="primary"):
        with st.spinner("Running health checks..."):
            all_passed = True
            for name, check_func in checks.items():
                with st.expander(f"{name} Check", expanded=True):
                    passed, message = check_func()
                    if passed:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                        all_passed = False
            
            if all_passed:
                st.success("🎉 All systems operational!")
            else:
                st.warning("⚠️ Some checks failed. Please review the details above.")

if __name__ == "__main__":
    main()