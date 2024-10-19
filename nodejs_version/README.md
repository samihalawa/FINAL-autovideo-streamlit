# Autovideo.ai

Autovideo.ai is a Progressive Web App (PWA) that generates videos using AI technologies.

## Features

- Generate video scripts using OpenAI GPT-3
- Create voiceovers using Hugging Face's text-to-speech model
- Fetch relevant video clips from YouTube
- Combine clips, voiceover, and background music
- Add titles and transitions to the final video
- PWA support for offline capabilities and easy installation
- Beautiful UI using Shadcn components
- Social sharing options
- Video preview functionality
- Responsive design for mobile and desktop

## Setup

1. Clone the repository
2. Install dependencies: `npm install`
3. Create a `.env` file with your API keys
4. Run setup script: `npm run setup`
5. Add background music files to `assets/music` directory
6. Start the server: `npm start`

## Usage

1. Open the app in your browser
2. Enter a video prompt, duration, and select a music style
3. Click "Create Video" and wait for the process to complete
4. Watch, share, and download your generated video

## Development

Run the app in development mode: `npm run dev`

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Environment Variables

Create a `.env` file in the root directory with the following variables:

## Deployment Instructions

1. Clone the repository:   ```
   git clone https://github.com/your-username/autovideo-ai.git
   cd autovideo-ai/nodejs_version   ```

2. Install dependencies:   ```
   npm install   ```

3. Set up environment variables:
   - Rename `.env.example` to `.env`
   - Replace the placeholder values with your actual API keys:     ```
     OPENAI_API_KEY=your_actual_openai_api_key
     HUGGINGFACE_API_KEY=your_actual_huggingface_api_key
     PORT=3000     ```

4. Create required directories:   ```
   mkdir -p assets/music output temp   ```

5. Add background music files to `assets/music` directory:
   - `upbeat.mp3`
   - `relaxing.mp3`
   - `dramatic.mp3`

6. Install FFmpeg:
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from https://ffmpeg.org/download.html and add to PATH

7. Start the server:   ```
   npm start   ```

8. Access the application at `http://localhost:3000`

## Troubleshooting

- If you encounter font-related issues, make sure the DejaVu Sans font is installed on your system, or update the `fontPath` variable in `videoGenerator.js` to point to an available font on your system.
- Ensure that you have sufficient disk space for temporary files and video output.
- Check the logs for any error messages and make sure all required API keys are correctly set in the `.env` file.
