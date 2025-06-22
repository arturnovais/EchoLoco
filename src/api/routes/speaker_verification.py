from fastapi import APIRouter, UploadFile, File
from api.schemas.speaker_verification import SpeakerResponse
from services.speaker_recognition.speaker_recognition import verify_speakers_cossine

router = APIRouter()

@router.post("/", response_model=SpeakerResponse)
async def speaker_identify(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    speaker = verify_speakers_cossine(temp_path)
    return SpeakerResponse(speaker=speaker)
