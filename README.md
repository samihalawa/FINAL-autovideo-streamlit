# Autovideo.ai

Autovideo.ai is a Progressive Web App (PWA) that allows users to create short videos based on text prompts. It uses AI to generate scripts, voiceovers, and images, combining them into a video.

## Features

1. Text-to-Video Generation: Create videos from text prompts.
2. AI-powered Script Generation: Automatically generate scripts based on user input.
3. Text-to-Speech Voiceover: Convert generated scripts into voiceovers.
4. AI Image Generation: Create relevant images for the video.
5. Video Composition: Combine generated elements into a final video.
6. Progress Tracking: Real-time updates on video creation progress.
7. Responsive Design: Works on desktop and mobile devices.
8. PWA Support: Can be installed as a standalone app on supported devices.
9. Download Option: Users can download their created videos.
10. Music Style Selection: Choose from different music styles for the video.

## Setup

1. Clone the repository
2. Install dependencies: `npm install`
3. Create a `.env` file with your API keys:   ```
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key   ```
4. Start the server: `npm start`
5. Open `http://localhost:3000` in your browser

## Technologies Used

- Node.js
- Express.js
- OpenAI API
- Hugging Face API
- FFmpeg
- Bull (for job queues)
- Tailwind CSS
- Alpine.js

## License

This project is licensed under the MIT License.





