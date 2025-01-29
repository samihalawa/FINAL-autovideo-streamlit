import { Button, Progress, Input, Select } from "@shadcn/ui";

const form = document.getElementById('videoForm');
const createVideoBtn = document.getElementById('createVideoBtn');
const promptInput = document.getElementById('prompt');
const durationInput = document.getElementById('duration');
const musicStyleSelect = document.getElementById('musicStyle');
const statusDiv = document.getElementById('status');
const videoDiv = document.getElementById('video');
const progressBar = document.getElementById('progressBar');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    createVideo();
});

async function createVideo() {
    try {
        createVideoBtn.disabled = true;
        statusDiv.textContent = 'Creating video...';
        videoDiv.innerHTML = '';
        progressBar.classList.remove('hidden');
        updateProgress(0);

        const formData = new FormData(form);
        const response = await fetch('/create-video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(Object.fromEntries(formData)),
        });

        if (!response.ok) {
            throw new Error('Server error');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const events = decoder.decode(value).split('\n\n');
            for (const event of events) {
                if (event.trim() === '') continue;
                const data = JSON.parse(event.replace('data: ', ''));
                
                if (data.progress) {
                    updateProgress(data.progress);
                } else if (data.videoUrl) {
                    statusDiv.textContent = 'Video created successfully!';
                    videoDiv.innerHTML = `<video class="w-full" controls><source src="${data.videoUrl}" type="video/mp4">Your browser does not support the video tag.</video>`;
                    addSharingOptions(data.videoUrl);
                } else if (data.error) {
                    throw new Error(data.error);
                }
            }
        }
    } catch (error) {
        statusDiv.textContent = 'Error: ' + error.message;
    } finally {
        createVideoBtn.disabled = false;
        progressBar.classList.add('hidden');
    }
}

function updateProgress(progress) {
    progressBar.style.width = progress + '%';
    progressBar.textContent = progress + '%';
}

function addSharingOptions(videoUrl) {
    const shareDiv = document.createElement('div');
    shareDiv.className = 'mt-4';
    shareDiv.innerHTML = `
        <h3 class="text-lg font-semibold mb-2">Share your video:</h3>
        <div class="flex space-x-2">
            <button onclick="shareOnTwitter('${videoUrl}')" class="bg-blue-400 text-white px-4 py-2 rounded">Twitter</button>
            <button onclick="shareOnFacebook('${videoUrl}')" class="bg-blue-600 text-white px-4 py-2 rounded">Facebook</button>
            <button onclick="copyToClipboard('${videoUrl}')" class="bg-gray-200 text-gray-800 px-4 py-2 rounded">Copy Link</button>
        </div>
    `;
    videoDiv.appendChild(shareDiv);

    const deleteButton = document.createElement('button');
    deleteButton.textContent = 'Delete Video';
    deleteButton.className = 'bg-red-500 text-white px-4 py-2 rounded';
    deleteButton.onclick = () => deleteVideo(videoUrl);
    shareDiv.appendChild(deleteButton);
}

window.shareOnTwitter = (url) => {
    window.open(`https://twitter.com/intent/tweet?url=${encodeURIComponent(url)}&text=Check out this video I created with Autovideo.ai!`, '_blank');
};

window.shareOnFacebook = (url) => {
    window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`, '_blank');
};

window.copyToClipboard = (url) => {
    navigator.clipboard.writeText(url).then(() => {
        alert('Video link copied to clipboard!');
    });
};

async function deleteVideo(videoUrl) {
    const filename = videoUrl.split('/').pop();
    try {
        const response = await fetch(`/delete-video/${filename}`, { method: 'DELETE' });
        if (response.ok) {
            alert('Video deleted successfully');
            videoDiv.innerHTML = '';
        } else {
            throw new Error('Failed to delete video');
        }
    } catch (error) {
        alert('Error deleting video: ' + error.message);
    }
}
