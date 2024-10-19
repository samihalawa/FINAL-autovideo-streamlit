// To run this script, use the command: node autovideo-js.js

import fs from 'fs';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegInstaller from '@ffmpeg-installer/ffmpeg';
import logger from './logger.js';
import axios from 'axios';
import { createCanvas } from 'canvas';
import OpenAI from 'openai';
import { fileURLToPath } from 'url';
import { LRUCache } from 'lru-cache';
import dotenv from 'dotenv';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Load environment variables
dotenv.config();

ffmpeg.setFfmpegPath(ffmpegInstaller.path);

const openai = new OpenAI();

const cache = new LRUCache({
  max: 100,
  ttl: 1000 * 60 * 60 * 24, // 24 hours
});

async function generateScript(prompt, duration) {
    if (!prompt || duration <= 0) {
        throw new Error('Invalid prompt or duration');
    }

    try {
        const response = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
                {role: "system", content: "You are a professional video script writer."},
                {role: "user", content: `Create a ${duration}-second video script about: ${prompt}. Include 8 scenes with detailed descriptions, durations, and suggested visuals. Each scene should be 5-15 seconds long.`}
            ],
            max_tokens: 1000,
        });
        return response.choices[0].message.content.trim();
    } catch (error) {
        logger.error('Error generating script:', error);
        throw new Error('Failed to generate script');
    }
}

async function generateVoiceover(script) {
    if (!script) {
        throw new Error('Invalid script');
    }

    try {
        const response = await axios.post(
            'https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits',
            { inputs: script },
            {
                headers: { Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}` },
                responseType: 'arraybuffer'
            }
        );

        const tempFile = path.join(__dirname, 'temp', `voiceover_${Date.now()}.wav`);
        fs.writeFileSync(tempFile, response.data);
        return tempFile;
    } catch (error) {
        logger.error('Error generating voiceover:', error);
        throw new Error('Failed to generate voiceover');
    }
}

async function fetchVideoClip(keywords, duration) {
    // Replace this function with a method that uses a different video source
    // or returns a default video clip
    return fetchDefaultVideoClip(duration);
}

async function createVideo(clips, voiceoverPath, outputPath, script, musicPath) {
    if (!clips || clips.length === 0 || !voiceoverPath || !outputPath || !script || !musicPath) {
        throw new Error('Invalid input parameters for video creation');
    }

    const fontPath = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';
    // If the above path doesn't work, you can use a fallback:
    // const fontPath = path.join(__dirname, 'assets', 'fonts', 'DejaVuSans-Bold.ttf');

    return new Promise((resolve, reject) => {
        const command = ffmpeg();

        clips.forEach(clip => {
            command.input(clip);
        });

        command.input(voiceoverPath);
        command.input(musicPath);

        const filterComplex = [
            ...clips.map((_, i) => `[${i}:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1[v${i}]`),
            ...clips.map((_, i) => i < clips.length - 1 ? `[v${i}][v${i+1}]xfade=transition=random:duration=1:offset=${10*(i+1)}[v${i+1}]` : '').filter(Boolean),
            `[v0]fade=t=in:st=0:d=1[v0]`,
            `[v${clips.length-1}]fade=t=out:st=${10*clips.length-1}:d=1[vout]`,
            `[${clips.length}:a]afade=t=in:st=0:d=1,afade=t=out:st=${10*clips.length-1}:d=1[a]`,
            `[${clips.length+1}:a]aloop=loop=-1:size=2s,afade=t=in:st=0:d=2,afade=t=out:st=${10*clips.length-2}:d=2,volume=0.1[music]`,
            `[a][music]amix=inputs=2[aout]`,
            `[vout]drawtext=fontfile=${fontPath}:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=h-th-10:text='${script.split('\n')[0]}':enable='between(t,0,5)',drawtext=fontfile=${fontPath}:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=h-th-50:text='Created by Autovideo.ai':enable='between(t,${10*clips.length-5},${10*clips.length})'[vtext]`
        ];

        command
            .complexFilter(filterComplex)
            .outputOptions(['-map', '[vtext]', '-map', '[aout]'])
            .videoCodec('libx264')
            .audioCodec('aac')
            .audioBitrate('192k')
            .videoBitrate('5000k')
            .size('1920x1080')
            .fps(30)
            .duration(10 * clips.length)
            .output(outputPath)
            .on('end', () => resolve(outputPath))
            .on('error', reject)
            .run();
    });
}

async function createVideoWorkflow(prompt, duration, musicStyle, jobId, progressCallback) {
    if (!prompt || duration <= 0 || !musicStyle || !jobId || typeof progressCallback !== 'function') {
        throw new Error('Invalid input parameters for video workflow');
    }

    const outputDir = path.join(__dirname, 'output');
    const tempDir = path.join(__dirname, 'temp');
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir);
    if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir);

    try {
        progressCallback(10);
        const script = await generateScript(prompt, duration);

        progressCallback(20);
        const voiceoverPath = await generateVoiceover(script);

        progressCallback(30);
        const musicPath = await fetchBackgroundMusic(musicStyle);

        progressCallback(40);
        const scenes = script.split('\n').filter(line => line.trim() !== '');
        const clips = [];
        for (let scene of scenes) {
            const keywords = scene.split(' ').slice(0, 5);
            const clipPath = await fetchVideoClip(keywords, 10);
            clips.push(clipPath);
        }

        progressCallback(60);
        const titlePath = await generateTitleImage(prompt, path.join(tempDir, `title_${jobId}.png`));
        clips.unshift(titlePath);

        progressCallback(80);
        const outputPath = path.join(outputDir, `${jobId}_output.mp4`);
        await createVideo(clips, voiceoverPath, outputPath, script, musicPath);

        progressCallback(100);
        return outputPath;
    } catch (error) {
        logger.error('Error in video creation workflow:', error);
        throw error;
    }
}

function cleanup() {
    const outputDir = path.join(__dirname, 'output');
    const tempDir = path.join(__dirname, 'temp');
    [outputDir, tempDir].forEach(dir => {
        const files = fs.readdirSync(dir);
        const now = Date.now();
        const oneDay = 24 * 60 * 60 * 1000;

        files.forEach(file => {
            const filePath = path.join(dir, file);
            const stats = fs.statSync(filePath);
            if (now - stats.mtimeMs > oneDay) {
                fs.unlinkSync(filePath);
            }
        });
    });
}

async function fetchBackgroundMusic(musicStyle) {
    if (!musicStyle) {
        throw new Error('Invalid music style');
    }

    const musicPath = path.join(__dirname, 'assets', 'music', `${musicStyle}.mp3`);
    if (!fs.existsSync(musicPath)) {
        logger.error(`No music found for style: ${musicStyle}`);
        // Fallback to a default music file
        const defaultMusicPath = path.join(__dirname, 'assets', 'music', 'default.mp3');
        if (!fs.existsSync(defaultMusicPath)) {
            throw new Error('No music files found. Please add music files to the assets/music directory.');
        }
        return defaultMusicPath;
    }
    return musicPath;
}

async function generateTitleImage(text, outputPath) {
    if (!text || !outputPath) {
        throw new Error('Invalid text or output path for title image');
    }

    const canvas = createCanvas(1920, 1080);
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, 1920, 1080);

    ctx.font = 'bold 60px sans-serif';
    ctx.fillStyle = 'white';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    ctx.fillText(text, 960, 540);

    const buffer = canvas.toBuffer('image/png');
    fs.writeFileSync(outputPath, buffer);

    return outputPath;
}

async function fetchDefaultVideoClip(duration) {
    const defaultClipPath = path.join(__dirname, 'assets', 'videos', 'default_clip.mp4');
    if (!fs.existsSync(defaultClipPath)) {
        throw new Error('Default video clip not found');
    }
    return defaultClipPath;
}

export { createVideoWorkflow, cleanup };
