from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from google.cloud import language_v1, texttospeech
from vertexai.generative_models import GenerativeModel
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import base64
import os
import io
from google.cloud import aiplatform, storage
import base64
from PIL import Image
from io import BytesIO
from google.cloud import storage
import google.generativeai as genai
import backoff
from google.api_core.exceptions import ResourceExhausted,GoogleAPICallError
import vertexai
import requests
import json
from fastapi.middleware.cors import CORSMiddleware
import bcrypt
from pydantic import BaseModel, EmailStr
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_DB_FILE = "users.json"

if not os.path.exists(USER_DB_FILE):
    with open(USER_DB_FILE, "w") as file:
        json.dump({}, file)

class RegisterUser(BaseModel):
    phone: str
    first_name: str
    last_name: str
    username: EmailStr
    password: str
    country: str


class LoginUser(BaseModel):
    username: EmailStr
    password: str

class User(BaseModel):
    username: str
    password: str

@app.get("/")
def read_root():
    return {"Error": "Please use the correct API endpoint"}

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_creds.json"
# Project ID and language configuration
project_id = "dogwood-flames-449418-n7"
language_code = "en-AU"
region = "us-central1"
endpoint_id=""
bucket_name = "hackthon_ai"

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
#endpoint = aiplatform.Endpoint(endpoint_name)

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
    #input_text = texttospeech.SynthesisInput(text=summary_text)
    input_text = texttospeech.SynthesisInput(
    ssml=f"""
    <speak>
        <prosody pitch="{1.0 if sentiment_score > 0.3 else (0.9 if sentiment_score < -0.3 else 0)}" rate="{0.9 if sentiment_score > 0.3 else (0.7 if sentiment_score < -0.3 else 1)}">
            {summary_text}
        </prosody>
    </speak>
    """
                )
    # Define voice parameters based on sentiment score
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, 
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # if sentiment_score > 0.3:  # Positive sentiment
    #     voice.pitch = 1.0  # Increase pitch slightly for enthusiasm
    #     voice.speaking_rate = 1.1  # Slightly faster pace
    # elif sentiment_score < -0.3:  # Negative sentiment
    #     voice.pitch = 0.9  # Slightly lower pitch for a more somber tone
    #     voice.speaking_rate = 0.9  # Slightly slower pace

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

    bucket_name = "hackthon_ai"
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
    
@backoff.on_exception(backoff.expo, [ResourceExhausted,GoogleAPICallError], max_tries=5)
def generate_prompt_image(commentary):
    vertexai.init(project=project_id, location=region)
    text1 = f"""You are a \"sports scene describer.\" Your task is to generate a highly detailed and realistic image prompt based on the provided live commentary of a sports match. The goal is to create an image prompt that can generate a photo-like representation of the live match, capturing the dynamic energy and atmosphere of the event while ensuring that faces are not generated in the final image.

Follow these steps:

1. **Extract Key Visual Elements:** Carefully analyze the live commentary below and identify all visible elements mentioned, including:

* **Players:** Their positions on the field, uniform details, body movements, and any specific actions (e.g., \"A player in a blue jersey dribbles past the defender,\" \"A forward leaps for a header\"). Avoid any mention of facial features or expressions.
* **Audience:** Describe the crowds presence, colors, and general movement without focusing on individual faces (e.g., \"A packed stadium of fans waving flags and raising their hands\").
* **Stadium Setting:** Include details of the stadium, such as architecture, lighting, advertisements, and scoreboard details.
* **Action Taking Place:** Precisely capture the movement of players and the ball (if applicable), ensuring a dynamic and realistic portrayal.
* **Weather Conditions:** Include relevant environmental factors such as lighting, temperature, or precipitation.
* **Specific Features:** Highlight any notable details like referee gestures, jerseys, goalposts, or player stances, but do not describe facial features.

2. **Craft a Detailed Image Prompt:** Use vivid and specific language to describe the scene while avoiding any facial details. Instead of mentioning faces or expressions, focus on body movements, postures, and interactions.

3. **Emphasize Realism:** Use keywords that promote photorealistic generation, such as \"photorealistic,\" \"high-definition,\" \"dynamic,\" \"action shot,\" \"live action,\" \"in the style of sports photography.\"

4. **Ensure Cohesion:** Integrate all elements into a single coherent scene, maintaining the overall atmosphere and energy of the match. Consider the best camera angles and perspectives (e.g., \"sideline view of a sprinting player,\" \"goalpost view of an incoming shot\").

5. **Explicitly Avoid Faces:** Include an instruction in the final prompt to ensure no faces are generated (e.g., \"The image should avoid generating faces or direct facial details, focusing instead on body movements, uniforms, and the dynamic energy of the game\").

6. **Output Format:** Format the image prompt as a single paragraph, ensuring clarity and coherence.

**Live Commentary:**
    {commentary}"""

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]
    vertexai.init(
        project="dogwood-flames-449418-n7",
        location="us-central1",
    )
    model = GenerativeModel(
        "gemini-1.5-pro-002",
    )
    responses = model.generate_content(
        [text1],
        generation_config=generation_config,
        safety_settings=safety_settings,
    
    )

    return str(responses.candidates[0].content.parts[0])



def generate_image_from_commentary(commentary: str):
    """
    Generate an image from the live commentary and save it to Google Cloud Storage.
    """
    try:
        import time
        time.sleep(60)
        prompt = generate_prompt_image(commentary)
        vertexai.init(project="dogwood-flames-449418-n7", location="us-central1")

        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

        images_gen = model.generate_images(
            prompt=prompt,
            # Optional parameters
            number_of_images=1,
            language="en",
            # You can't use a seed value and watermark at the same time.
            add_watermark=False,
            guidance_scale = 0.9,
            seed=100,
            aspect_ratio="1:1",
            safety_filter_level="BLOCK_NONE")
        print(prompt,images_gen,"/n/n/n\n\n")

        # Parse the response
        if images_gen:
            if images_gen.images:
                # Decode and save the image locally

                generated_image = images_gen.images[0]  # This is a GeneratedImage object
                print(type(generated_image),"hnlooo")
                local_file_path = "./gen_images/generated_image.png"
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                with open(local_file_path, "wb") as f:
                    generated_image.save(f)  # Correct way to save the image
                
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
        print("endpointwwfebewfbewfbewbewbfew")
        key_moment = analyze_key_moment(request.fav_team, request.fav_player, request.commentary)
        if key_moment.lower() != "yes":
            return {"key_moment": "No", "message": "The commentary does not contain a key moment."}
        print("key_moemnet_done")
        # Step 2: Sentiment analysis
        sentiment_score = analyze_sentiment(request.commentary)
        print("sentiment_score_done")
        # Step 3: Generate summary
        summary = generate_summary(request.commentary, sentiment_score)
        print("summary_done")
        # Step 4: Generate audio
        audio = set_tone_by_sentiment(request.fav_player,sentiment_score, summary, request.language_code)
        print("audio_done")
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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    

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
    """Load users from a JSON file"""
    try:
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_users(users):
    """Save users to a JSON file"""
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against the stored hash"""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


@app.post("/register")
def register(user: RegisterUser):
    users = load_users()

    if user.username in users:
        raise HTTPException(status_code=400, detail="User already exists. Please login.")

    users[user.username] = {
        "phone": user.phone,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "password": hash_password(user.password),
        "country": user.country,
    }
    
    save_users(users)
    return {"message": f"User {user.first_name} registered successfully!"}


@app.post("/login")
def login(user: LoginUser):
    users = load_users()

    if user.username not in users:
        raise HTTPException(status_code=400, detail="User not found. Please register.")

    stored_password = users[user.username]["password"]
    
    if verify_password(user.password, stored_password):
        return {"message": f"Welcome back, {users[user.username]['first_name']}!"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials. Please try again.")


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
        print(PlayerData)
        try:
            prompt = f"""Generate a highly realistic and detailed image of a baseball scorecard displaying player statistics.

            The scorecard should clearly showcase the following information:

            * **Player Name**: {data.get("fullname", "")}
            * **Debut Date**: {data.get("mlbDebutDate", "")}
            * **Height**: {data.get("height", "")}
            * **Weight**: {data.get("weight", "")}
            * **Age**: {data.get("currentAge", "")}
            * **Batting Side**: {data.get("batSide", "")}
            * **Strike Zone**: Top - {data.get("strikeZoneTop", "")}, Bottom - {data.get("strikeZoneBottom", "")}
            * **Pitching Hand**: {data.get("pitchHand", "")}

            The scorecard should have a professional, clean, and realistic design, resembling an official baseball stat sheet.  It should be placed on a visually appealing background, such as a wooden table, a baseball dugout, or a scoreboard, to add depth to the scene.  The overall image should evoke the atmosphere of a professional baseball game.

            Ensure the scorecard is the central focus of the image.  Do not include any faces, portraits, or human figures.  Focus solely on the scorecard and related elements, such as baseball equipment (bats, gloves, balls), or elements of the chosen background setting.

            Style: Photorealistic, high detail, professional sports photography.

            Lighting: Natural, slightly dramatic lighting to enhance the realism and depth of the scene."""
            vertexai.init(project="dogwood-flames-449418-n7", location="us-central1")

            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

            images_gen = model.generate_images(
                prompt=prompt,
                # Optional parameters
                number_of_images=1,
                language="en",
                # You can't use a seed value and watermark at the same time.
                add_watermark=False,
                guidance_scale = 0.9,
                seed=100,
                aspect_ratio="1:1",
                safety_filter_level="BLOCK_NONE")
            if images_gen:
                    if images_gen.images:
                        # Decode and save the image locally
                        generated_image = images_gen.images[0]  # This is a GeneratedImage object
                        print(type(generated_image),"hnlooo")
                        local_file_path = "./gen_images/generated_image.png"
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        with open(local_file_path, "wb") as f:
                            generated_image.save(f)  # Correct way to save the image
                        print(f"Image generated and saved locally as '{local_file_path}'.")
                        # Upload the image to GCS
                        file_name = "generated_images/generated_image.png"  # Path in GCS bucket
                        url_image = upload_to_gcs(local_file_path, bucket_name, file_name)
                        return {"player_data": PlayerData,"image_url":url_image} 
        except Exception as e:
            print(f"An error occurred: {e}")
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
def generate_image(request: PlayerData) -> Dict:
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