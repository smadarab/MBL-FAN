from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from google.cloud import language_v1, texttospeech
from vertexai.generative_models import GenerativeModel
import base64
import os
from google.cloud import storage
import vertexai
import requests
import json

# FastAPI app instance
app = FastAPI()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_creds.json"
# Project ID and language configuration
project_id = "red-seeker-447614-t6"
language_code = "en-AU"

# Input schema for API endpoints
class CommentaryRequest(BaseModel):
    fav_team: str
    fav_player: str
    commentary: str
    language_code : str

class IdRequest(BaseModel):
    team_id : int
    player_id: int

# VertexAI initialization
def init_vertexai():
    vertexai.init(project=project_id, location="us-central1")

def analyze_key_moment(fav_team: str, fav_player: str, commentary: str) -> str:    
    init_vertexai()
    user_prompt = f"""Based on the following preferences, classify whether this commentary snippet contains a key moment:

        - Favorite Team: [Team Name]
        - Favorite Player: [Player Name]
        - Selectable Factors: [e.g., Scoring milestones, fouls, three-pointers, game-changing plays, etc.]

        Commentary Snippet: \"[Insert commentary snippet]\"

        Does this contain a key moment? Respond with \"Yes\" or \"No.\"

        input: Based on the following preferences, classify whether this commentary snippet contains a key moment:

        - Favorite Team: Los Angeles Lakers
        - Favorite Player: LeBron James
        - Selectable Factors: Scoring milestones, three-pointers, game-changing plays

        Commentary Snippet: \"LeBron James hits a stunning three-pointer to put the Lakers ahead by 5!\"

        Does this contain a key moment? Respond with \"Yes\" or \"No.\"
        output: Yes

        input: Based on the following preferences, classify whether this commentary snippet contains a key moment:

        - Favorite Team: Los Angeles Lakers
        - Favorite Player: LeBron James
        - Selectable Factors: Scoring milestones, three-pointers, game-changing plays

        Commentary Snippet: \"Warriors' Stephen Curry sinks another three-pointer, narrowing the gap to 2 points.\"

        Does this contain a key moment? Respond with \"Yes\" or \"No.\"

        output: No



        input: Based on the following preferences, classify whether this commentary snippet contains a key moment:

        - Favorite Team: {fav_team}
        - Favorite Player: {fav_player}
        - Selectable Factors: Scoring milestones, three-pointers, game-changing plays

        Commentary Snippet: {commentary}
        Does this contain a key moment? Respond with \"Yes\" or \"No.\"

        output:"""

    system_prompt = """You are an AI model that analyzes live basketball commentary and determines if it contains any key moments based on user-defined preferences. You must classify each commentary snippet as \"Yes\" (contains a key moment) or \"No\" (does not contain a key moment) based on the provided preferences such as favorite team, favorite player, and selectable factors (e.g., scoring milestones, fouls, three-pointers, game-changing plays). Respond with only \"Yes\" or \"No.\""""

    generation_config = {
        "max_output_tokens": 100,
        "temperature": 0,
        "top_p": 0.95,
    }
    model = GenerativeModel(
            "gemini-1.5-pro-002",
            system_instruction=[system_prompt]
        )
    responses = model.generate_content(
            [user_prompt],
            generation_config=generation_config,
            stream=True,
        )
    return list(responses)[0].text

def analyze_sentiment(text):
    language_client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = language_client.analyze_sentiment(document=document).document_sentiment
    return sentiment.score

def generate_summary(commentary: str, sentiment_score: float) -> str:
    init_vertexai()
    user_prompt = f"""Input Commentary Feed: {commentary}
    Sentiment Score: {sentiment_score}
    Generate a concise one-line summary that reflects the key event and incorporates the sentiment tone."""
    system_prompt = """You are an AI model that summarizes sports commentary into concise single-line summaries while reflecting sentiment tone."""
    generation_config = {"max_output_tokens": 8192, "temperature": 1, "top_p": 0.95}
    model = GenerativeModel("gemini-1.5-pro-002", system_instruction=[system_prompt])
    responses = model.generate_content([user_prompt], generation_config=generation_config)
    return responses.text.strip()

# Function to generate audio with tone based on sentiment
def set_tone_by_sentiment(player:str,sentiment_score: float, summary_text: str, language_code: str) -> str:
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=summary_text)

    # Define voice parameters based on sentiment score
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, 
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    if sentiment_score > 0.3:  # Positive sentiment
        voice.pitch = 1.0  # Increase pitch slightly for enthusiasm
        voice.speaking_rate = 1.1  # Slightly faster pace
    elif sentiment_score < -0.3:  # Negative sentiment
        voice.pitch = 0.9  # Slightly lower pitch for a more somber tone
        voice.speaking_rate = 0.9  # Slightly slower pace

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    file_name = f"{player}.mp3"
    local_file_path = f"{file_name}"

    # Save MP3 locally
    with open(local_file_path, "wb") as out:
        out.write(response.audio_content)

    bucket_name = "hackathonbucket" 
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(local_file_path)
    blob.make_public()
    return blob.public_url   

# API endpoint to process commentary
@app.post("/process-commentary")
def process_commentary(request: CommentaryRequest) -> Dict:
    try:
        # Step 1: Analyze if it's a key moment
        key_moment = analyze_key_moment(request.fav_team, request.fav_player, request.commentary)
        if key_moment.lower() != "yes":
            return {"key_moment": "No", "message": "The commentary does not contain a key moment."}

        # Step 2: Sentiment analysis
        sentiment_score = analyze_sentiment(request.commentary)

        # Step 3: Generate summary
        summary = generate_summary(request.commentary, sentiment_score)

        # Step 4: Generate audio
        audio = set_tone_by_sentiment(request.fav_player,sentiment_score, summary, request.language_code)

        return {
            "key_moment": key_moment.lower(),
            "sentiment_score": sentiment_score,
            "summary": summary,
            "audio": audio,  # Audio encoded in base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/team/{team_id}")
def get_team_data(team_id : int):
    try:
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}"
        response = requests.get(url)
        response.raise_for_status()
        team_data = response.json()
        return {"team_data": team_data}
    except:
        return {"error": "Invalid team ID"}

@app.get("/player_id/{id}")
def get_player_data(id: int):
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{id}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        player_data = response.json()
        data = player_data.get('people')
        
        if not data or not isinstance(data, list):
            raise HTTPException(status_code=404, detail="Player data not found")
        
        player = data[0]
        result = {
            "fullname": player.get("fullName", "Unknown"),
            "pitchHand": player.get("pitchHand", {}),
            "batSide": player.get("batSide", {})
        }
        
        return {"player_data": result} 
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except KeyError as e:
        # Handle missing keys in the JSON response
        raise HTTPException(status_code=500, detail=f"Data parsing error: Missing key {str(e)}")
    except Exception as e:
        # Catch-all for other exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")