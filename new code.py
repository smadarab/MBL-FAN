import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from google.cloud import language_v1, texttospeech, vision
from google.cloud import texttospeech
from google import generativeai
from google.generativeai import *
from google.generativeai import types
import base64
project_id = "replace your project code here"

fav_team,fav_player,commentry = "Golden State Warriors","Stephen Curry","Stephen Curry hits a game-winning three-pointer from 30 feet out in the final seconds of the game"
language_code= "en-AU"


def generate(fav_team,fav_player,commentry):
    vertexai.init(project=project_id, location="us-central1")
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

        Commentary Snippet: {commentry}
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

def set_tone_by_sentiment(sentiment_score, summary_text,language_code):
    """
    Sets the tone of the synthesized audio based on the sentiment score.

    Args:
        sentiment_score: The sentiment score of the text, ranging from -1 (very negative) to 1 (very positive).
        summary_text: The text to be synthesized into audio.

    Returns:
        The synthesized audio content.
    """

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
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
    return response.audio_content

def analyze_sentiment(text, language_client):
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = language_client.analyze_sentiment(document=document).document_sentiment
    return sentiment.score



def generate_summary(commentary,sentiment_score):
        generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}
        text1 = f"""Here is a new commentary feed with a sentiment score:
                    Input Commentary Feed: [Insert single-line commentary feed here]
                    Sentiment Score: [Insert sentiment score here]
                    Based on the examples provided, generate a concise one-line summary that reflects the key event and incorporates the sentiment tone

                    input: Input Commentary Feed:
                    \"Spectacular home run by the rookie! The crowd is going wild!\"
                    Sentiment Score: 0.95
                    output: The rookie's incredible home run electrified the crowd with excitement.

                    input: Input Commentary Feed:
                    \"The pitcher is really struggling to find the strike zone today—three walks in a row.\"
                    Sentiment Score: -0.6
                    output: The pitcher's control issues led to three consecutive walks, creating tension.


                    input: Input Commentary Feed: {commentary}
                    Sentiment Score: {sentiment_score}
                    output:"""
        textsi_1 = """You are an advanced language model specializing in generating concise summaries for single-line baseball commentary. Your task is to summarize the provided commentary in a single sentence, incorporating the sentiment score to reflect the tone appropriately. Keep the summary short and focused"""

        model = GenerativeModel(
        "gemini-1.5-pro-002",
        system_instruction=[textsi_1]
    )
        responses = model.generate_content(
        [text1],
        generation_config=generation_config,
        
    )

        return responses.text


key_moment = generate(fav_team,fav_player,commentry)
if key_moment.lower()=="yes":
    language_client = language_v1.LanguageServiceClient()
    sentiment_score = analyze_sentiment(commentry,language_client)
    """Sentiment Score Range
            -1.0 to -0.25: Negative sentiment
            Indicates the text expresses negative emotions, dissatisfaction, or criticism.

            -0.25 to 0.25: Neutral sentiment
            Indicates the text does not express strong emotions, being more factual or balanced.

            0.25 to 1.0: Positive sentiment
            Indicates the text expresses positive emotions, satisfaction, or praise."""
    summarised_content = generate_summary(commentry,sentiment_score)
    audio = set_tone_by_sentiment(sentiment_score,summarised_content,language_code) 
else:
    print("Not a keymoment")    