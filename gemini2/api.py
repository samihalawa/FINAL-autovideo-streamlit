from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from app import process_video_generation

class VideoRequest(BaseModel):
    topic: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Video Generation API is running"}

@app.post("/generate")
async def generate_video(request: VideoRequest):
    try:
        output_path = await process_video_generation(request.topic)
        return {
            "status": "success",
            "message": "Video generated successfully",
            "video_path": output_path
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 