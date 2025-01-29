import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { createVideoWorkflow, cleanup } from './videoGenerator.js';
import logger from './logger.js';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import { LRUCache } from 'lru-cache';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const requiredDirs = ['assets/music', 'output', 'temp'];

requiredDirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
        console.error(`Error: ${dir} directory does not exist. Run 'npm run setup' to create required directories.`);
        process.exit(1);
    }
});

const app = express();
app.use(express.json());
app.use(express.static('public'));
app.use('/output', express.static(path.join(__dirname, 'output')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const cache = new LRUCache({
  max: 100,
  ttl: 1000 * 60 * 60 * 24, // 24 hours
});

app.post('/create-video', async (req, res) => {
    const { prompt, duration, musicStyle } = req.body;
    const jobId = uuidv4();

    if (!prompt || !duration || !musicStyle) {
        return res.status(400).json({ error: 'Missing required fields' });
    }

    if (isNaN(duration) || duration <= 0 || duration > 300) {
        return res.status(400).json({ error: 'Duration must be between 1 and 300 seconds' });
    }

    res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    });

    try {
        const cacheKey = `${prompt}-${duration}-${musicStyle}`;
        if (cache.has(cacheKey)) {
            const cachedVideoFile = cache.get(cacheKey);
            res.write(`data: ${JSON.stringify({ videoUrl: `/output/${path.basename(cachedVideoFile)}` })}\n\n`);
            res.end();
            return;
        }

        const videoFile = await createVideoWorkflow(prompt, parseInt(duration), musicStyle, jobId, (progress) => {
            res.write(`data: ${JSON.stringify({ progress })}\n\n`);
        });
        cache.set(cacheKey, videoFile);
        res.write(`data: ${JSON.stringify({ videoUrl: `/output/${path.basename(videoFile)}` })}\n\n`);
    } catch (error) {
        logger.error('Error in video creation:', error);
        res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
    } finally {
        res.end();
    }
});

app.get('/manifest.json', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'manifest.json'));
});

app.get('/sw.js', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'sw.js'));
});

// Add error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});

// Run cleanup every 24 hours
setInterval(cleanup, 24 * 60 * 60 * 1000);

// Add this new route
app.delete('/delete-video/:filename', (req, res) => {
    const filename = req.params.filename;
    const filePath = path.join(__dirname, 'output', filename);

    fs.unlink(filePath, (err) => {
        if (err) {
            logger.error(`Error deleting file: ${err}`);
            return res.status(500).json({ error: 'Failed to delete video' });
        }
        res.json({ message: 'Video deleted successfully' });
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
    console.log('Make sure to set up your .env file with the required API keys before running the server.');
});
