from dataclasses import asdict
import logging
import tempfile
from uu import decode

from .utils import split_on_pause, split_on_nothing, join_if_short
from .config import user_setting_path
from .core import DecodeConfig, Middleware, Voice, download
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, APIRouter
import json
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse
import wave


voicecraft_path, encodec_path = download()
m = Middleware()
m.load(voicecraft_path, encodec_path)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


class TTSSettingsRequest(BaseModel):
    stream_chunk_size: int
    temperature: float
    speed: float
    length_penalty: float
    repetition_penalty: float
    top_p: float
    top_k: int
    enable_text_splitting: bool


class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: str
    language: str


@app.get("/")
async def root():
    return {"message": "VoiceCraftST"}


@app.get("/health")
async def root():
    return {"message": "Does it work? Who knows. But the API is up!"}


@app.get("/speakers")
@router.get("/speakers")
def get_speakers():
    return list(map(lambda v: v.describe(), Voice.get_voices().values()))


@router.post("/tts_to_audio")
@router.post("/tts_to_audio/")
def tts_to_audio(user: str, stop_repetition: int, sample_batch_size: int, request: SynthesisRequest):
    enable_text_splitting = True

    if user_setting_path(user):
        with open(user_setting_path(user), "r") as f:
            data = json.load(f)
            decode_config = DecodeConfig(
                top_k=data["top_k"],
                top_p=data["top_p"],
                temperature=data["temperature"],
            )
            enable_text_splitting = data["enable_text_splitting"]
    else:
        decode_config = DecodeConfig()

    decode_config.stop_repetition = stop_repetition
    decode_config.sample_batch_size = sample_batch_size

    transcript = request.text.replace("*", "").replace('"', "")

    output_file_path = m.generate(
        Voice.get_voices()[request.speaker_wav],
        transcript,
        decode_config,
        True,
        (lambda x: join_if_short(split_on_pause(x), 80)) if enable_text_splitting else split_on_nothing,
    )

    return FileResponse(
        path=output_file_path,
        media_type="audio/wav",
        filename="output.wav",
    )


@router.post("/set_tts_settings")
def set_settings(user: str, tts_settings_req: TTSSettingsRequest):
    with open(os.path.join("/users", f"{user}.json"), "w") as f:
        json.dump(
            {
                "stream_chunk_size": tts_settings_req.stream_chunk_size,
                "temperature": tts_settings_req.temperature,
                "speed": tts_settings_req.speed,
                "length_penalty": tts_settings_req.length_penalty,
                "repetition_penalty": tts_settings_req.repetition_penalty,
                "top_p": tts_settings_req.top_p,
                "top_k": tts_settings_req.top_k,
                "enable_text_splitting": tts_settings_req.enable_text_splitting,
            },
            f,
        )
    return tts_settings_req


@app.post("/add_speaker")
async def add_speaker(name=Form(...), transcript=Form(...), file: UploadFile = File(...)):
    print(
        f"Received request to add speaker, content: {file.content_type}, file: {file.filename}, name: {name}, transcript: {transcript}"
    )

    if file.size and file.size > 2e9:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Max {2e9} bytes")

    with tempfile.NamedTemporaryFile() as f:
        # https://medium.com/@jayhawk24/upload-files-in-fastapi-with-file-validation-787bd1a57658
        # this seems all completely stupid
        real_file_size = 0
        for chunk in file.file:
            real_file_size += len(chunk)
            if real_file_size > 2e9:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Max {2e9} bytes")
            f.write(chunk)

        with wave.open(f.name, "rb") as wave_file:
            frame_rate = wave_file.getframerate()
            if frame_rate != 16e3:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Model works only with 16kHz wav files"
                )

        voice = Voice.from_sample(name, f.name, transcript)

    print(f"Added voice with id {voice.id}")
    return voice.describe()


@router.get("/get_tts_settings")
def get_tts_settings(user: str):
    if not user_setting_path(user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No config for this user")

    with open(user_setting_path(user), "r") as f:
        return json.load(f)


app.include_router(router, prefix="/{user}/{stop_repetition}/{sample_batch_size}")
