from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from google.cloud import language_v1, texttospeech
from vertexai.generative_models import GenerativeModel
import base64
import os
from google.cloud import aiplatform, storage
import base64
from PIL import Image
from io import BytesIO
import os
from google.cloud import storage
import vertexai
import requests
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define the file where user data will be stored
USER_DATA_FILE = "users.json"

# Ensure the file exists
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "w") as file:
        json.dump({}, file)

class User(BaseModel):
    username: str
    password: str

@app.get("/")
def read_root():
    return {"Error": "Please use the correct API endpoint"}

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_creds.json"
# Project ID and language configuration
project_id = "add_project_id"
language_code = "en-AU"
region = "asia-east1"
endpoint_id = "endpoint"
bucket_name = "bucketname"

# class PlayerData(BaseModel):
#     fullname: str
#     pitchHand: dict
#     batSide: dict
#     strikeZoneTop: dict
#     strikeZoneBottom: dict
#     height: dict
#     weight: dict
#     mlbDebutDate: dict
#     currentAge: dict
#     birthDate: dict

class PlayerData(BaseModel):
    playerdata : dict

# Initialize GCP AI Platform
aiplatform.init(project=project_id, location=region)
# Construct the endpoint resource name
endpoint_name = f"projects/{project_id}/locations/{region}/endpoints/{endpoint_id}"
# Initialize the endpoint
endpoint = aiplatform.Endpoint(endpoint_name)

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

def create_prompt(commentary: str) -> str:
    """
    Generate a descriptive prompt based on the live commentary.
    """
    return f"Create a realistic image of {commentary.lower()}."


def create_player_prompt(data):
    """
    Generate a descriptive prompt based on the live commentary.
    """
    res = f"""Create a realistic image of score card  with player name as {data.get("fullname", "")} with debut date as {data.get("mlbDebutDate", {})} whose height is {data.get("height", "")} 
    and weight is {data.get("weight", "")} and age is {data.get("currentAge", "")} with batSide as {data.get("batSide", {})} , strikeZoneTop as {data.get("strikeZoneTop", {})} , strikeZoneBottom as {data.get("strikeZoneBottom", {})} and pitchHand as {data.get("pitchHand", {})}."""
    print(res)

    return f"""Create a realistic image of score card  with player name as {data.get("fullname", "")} with debut date as {data.get("mlbDebutDate", {})} whose height is {data.get("height", "")} 
    and weight is {data.get("weight", "")} and age is {data.get("currentAge", "")}
    with batSide as {data.get("batSide", {})} , strikeZoneTop as {data.get("strikeZoneTop", {})} , strikeZoneBottom as {data.get("strikeZoneBottom", {})} and pitchHand as {data.get("pitchHand", {})}  ."""

def upload_to_gcs(local_file_path: str, bucket_name: str, file_name: str) -> str:
    """
    Upload a file to Google Cloud Storage and return its public URL.
    """
    try:
        # Initialize GCS client
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Upload file to GCS
        blob.upload_from_filename(local_file_path)
        blob.make_public()
        public_url = blob.public_url
        print(f"File uploaded to GCS bucket '{bucket_name}' with public URL: {public_url}")
        return public_url
    except Exception as e:
        print(f"Failed to upload file to GCS: {e}")
        return ""

def generate_image_from_commentary(commentary: str):
    """
    Generate an image from the live commentary and save it to Google Cloud Storage.
    """
    prompt = create_prompt(commentary)
    instances = [
        prompt
    ]

    try:
        # Make the prediction request
        response = endpoint.predict(instances=instances, parameters={
            "guidance_scale": 0.1,
            "negative_prompt": "cartoonish, unrealistic, overexposed, distorted faces, blurry, flat colors",
            "num_inference_steps": 200,
            "width": 768,
            "height": 768,
            "seed": 12345
        })

        # Parse the response
        if response.predictions:
            # Expecting the first prediction to be a base64-encoded string
            generated_image = response.predictions[0] if isinstance(response.predictions[0], str) else None

            if generated_image:
                # Decode and save the image locally
                image_data = base64.b64decode(generated_image)
                image = Image.open(BytesIO(image_data))
                local_file_path = "./gen_images/generated_image.png"
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                image.save(local_file_path)
                print(f"Image generated and saved locally as '{local_file_path}'.")

                # Upload the image to GCS
                file_name = "generated_images/generated_image.png"  # Path in GCS bucket
                url_image = upload_to_gcs(local_file_path, bucket_name, file_name)
                return url_image
            else:
                print("No valid image data found in response.")
        else:
            print("No predictions returned by the model.")
    except Exception as e:
        print(f"An error occurred: {e}")


from datetime import datetime

def convert_player_data(PlayerData):
    # Convert player data to the required format
    converted_data = {
        "fullname": PlayerData["fullname"],
        "pitchHand": {
            "type": PlayerData["pitchHand"]["description"]
        },
        "batSide": {
            "type": PlayerData["batSide"]["description"]
        },
        "strikeZoneTop": {
            "value": PlayerData["strikeZoneTop"]
        },
        "strikeZoneBottom": {
            "value": PlayerData["strikeZoneBottom"]
        },
        "height": {
            "value": PlayerData["height"]
        },
        "weight": {
            "value": str(PlayerData["weight"]) + "lbs"  # Add lbs to weight
        },
        "mlbDebutDate": {
            "date": PlayerData["mlbDebutDate"]
        },
        "currentAge": {
            "value": PlayerData["currentAge"]
        },
        "birthDate": {
            "date": PlayerData["birthDate"]
        }
    }
    return converted_data

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
        image_url = generate_image_from_commentary(request.commentary)
        return {
            "key_moment": key_moment.lower(),
            "sentiment_score": sentiment_score,
            "summary": summary,
            "audio": audio,  # Audio encoded in base64
            "image":image_url
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


def load_users():
    """Load users from the JSON file."""
    with open(USER_DATA_FILE, "r") as file:
        return json.load(file)

def save_users(users):
    """Save users to the JSON file."""
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users, file, indent=4)

@app.post("/login")
def login(user: User):
    users = load_users()

    if user.username in users:
        if users[user.username] == user.password:
            return {"message": f"Welcome back! {user.username}"}
        else:
            return {"message": f"Welcome {user.username}, but password updated!"}
    
    # If username is new, store it
    users[user.username] = user.password
    save_users(users)
    return {"message": f"Welcome {user.username}"}


@app.get("/team_players/{id}")
def get_team_players(id: int):
    try:
        # Fetch data from the external API
        url = f"https://statsapi.mlb.com/api/v1/teams/{id}/roster?season=2024"
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the response JSON
        data = response.json()
        
        # Extract the players' id and fullName
        players = [
            {
                "id": player["person"]["id"],
                "fullName": player["person"]["fullName"]
            }
            for player in data.get("roster", [])
        ]
        
        return {"team_id": id, "players": players}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {e}")

    except KeyError:
        raise HTTPException(status_code=500, detail="Unexpected response structure from API")

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
        PlayerData = {
            "fullname": player.get("fullName", "Unknown"),
            "pitchHand": player.get("pitchHand", {}),
            "batSide": player.get("batSide", {}),
            "strikeZoneTop": player.get("strikeZoneTop", {}),
            "strikeZoneBottom": player.get("strikeZoneBottom", {}),
            "height": player.get("height", {}),
            "weight": player.get("weight", {}),
            "mlbDebutDate" : player.get("mlbDebutDate", {}),
            "currentAge": player.get("currentAge", {}),
            "birthDate": player.get("birthDate", {})
        }
        # data = convert_player_data(PlayerData)
        # data = convert_player_data(PlayerData)
        # data = json.dumps(PlayerData)
        # print(PlayerData)
        # prompt = create_player_prompt(PlayerData)

        return {"player_data": PlayerData} 
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except KeyError as e:
        # Handle missing keys in the JSON response
        raise HTTPException(status_code=500, detail=f"Data parsing error: Missing key {str(e)}")
    except Exception as e:
        # Catch-all for other exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/generate_image")
def generate_image_from_commentary(request: PlayerData) -> Dict:
    """
    Generate an image from the live commentary and save it to Google Cloud Storage.
    """
    # data = convert_player_data(PlayerData)
    # data = json.dumps(PlayerData)
    prompt = create_player_prompt(PlayerData)
    instances = [
        prompt
    ]

    try:
        # Make the prediction request
        response = endpoint.predict(instances=instances, parameters={
            "guidance_scale": 0.1,
            "negative_prompt": "cartoonish, unrealistic, overexposed, distorted faces, blurry, flat colors",
            "num_inference_steps": 200,
            "width": 768,
            "height": 768,
            "seed": 12345
        })

        # Parse the response
        if response.predictions:
            # Expecting the first prediction to be a base64-encoded string
            generated_image = response.predictions[0] if isinstance(response.predictions[0], str) else None

            if generated_image:
                # Decode and save the image locally
                image_data = base64.b64decode(generated_image)
                image = Image.open(BytesIO(image_data))
                local_file_path = "./gen_images/player_image.png"
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                image.save(local_file_path)
                print(f"Image generated and saved locally as '{local_file_path}'.")

                return image
            else:
                print("No valid image data found in response.")
        else:
            print("No predictions returned by the model.")
    except Exception as e:
        print(f"An error occurred: {e}")