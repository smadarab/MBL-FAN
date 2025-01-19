from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from google.cloud import language_v1, texttospeech
from vertexai.generative_models import GenerativeModel
import vertexai
import os

# FastAPI app
app = FastAPI()

# Set Google Cloud project ID (Replace with your project code)
project_id = "replace your project code here"

# Initialize Vertex AI
vertexai.init(project=project_id, location="us-central1")

class CommentaryRequest(BaseModel):
    fav_team: str
    fav_player: str
    commentary: str
    language_code: Optional[str] = "en-AU"

class CommentaryResponse(BaseModel):
    key_moment: bool
    sentiment_score: Optional[float] = None
    summary: Optional[str] = None
    audio_file: Optional[str] = None


# Helper functions
def generate_key_moment(fav_team, fav_player, commentary):
    user_prompt = f"""Based on the following preferences, classify whether this commentary snippet contains a key moment:

        - Favorite Team: {fav_team}
        - Favorite Player: {fav_player}
        - Selectable Factors: Scoring milestones, three-pointers, game-changing plays

        Commentary Snippet: {commentary}
        Does this contain a key moment? Respond with \"Yes\" or \"No.\"

        output:"""

    system_prompt = """You are an AI model that analyzes live basketball commentary and determines if it contains any key moments."""
    generation_config = {"max_output_tokens": 100, "temperature": 0, "top_p": 0.95}
    
    model = GenerativeModel("gemini-1.5-pro-002", system_instruction=[system_prompt])
    responses = model.generate_content([user_prompt], generation_config=generation_config, stream=True)
    return list(responses)[0].text.strip().lower() == "yes"


def analyze_sentiment(text):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    return sentiment.score


def generate_summary(commentary, sentiment_score):
    user_prompt = f"""Generate a concise one-line summary for the following commentary:
    
        Commentary: {commentary}
        Sentiment Score: {sentiment_score}
        
        output:"""
    
    system_prompt = """You are an advanced language model specializing in generating concise summaries for basketball commentary."""
    generation_config = {"max_output_tokens": 100, "temperature": 0.7, "top_p": 0.9}
    
    model = GenerativeModel("gemini-1.5-pro-002", system_instruction=[system_prompt])
    responses = model.generate_content([user_prompt], generation_config=generation_config, stream=True)
    return list(responses)[0].text.strip()


def set_tone_by_sentiment(sentiment_score, summary_text, language_code):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=summary_text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    audio_path = "output.mp3"
    with open(audio_path, "wb") as audio_file:
        audio_file.write(response.audio_content)
    return audio_path


@app.post("/analyze_commentary", response_model=CommentaryResponse)
async def analyze_commentary(request: CommentaryRequest):
    try:
        is_key_moment = generate_key_moment(request.fav_team, request.fav_player, request.commentary)
        if not is_key_moment:
            return CommentaryResponse(key_moment=False)

        sentiment_score = analyze_sentiment(request.commentary)
        summary = generate_summary(request.commentary, sentiment_score)
        audio_path = set_tone_by_sentiment(sentiment_score, summary, request.language_code)

        return CommentaryResponse(
            key_moment=True,
            sentiment_score=sentiment_score,
            summary=summary,
            audio_file=audio_path
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
