import os
from dotenv import load_dotenv
import openai

def gpt_translate(text, source_lang, target_lang):
    """
    Translates text from source_lang to target_lang using OpenAI GPT-4.1 API.
    Loads the API key from a .env file.
    Args:
        text (str): The text to translate.
        source_lang (str): The source language code (e.g., 'en').
        target_lang (str): The target language code (e.g., 'fr').
    Returns:
        str: The translated text.
    """
    # Load environment variables from .env file
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    client = openai.OpenAI()

    # Construct the prompt for translation using language codes
    prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}"

    # Call the OpenAI API for translation
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "user", "content": prompt}
        ],
    )

    # Extract the translated text from the response
    translated_text = response.output_text

    return translated_text


# test the function
if __name__ == "__main__":
    text = "Hello, how are you?"
    source_lang = "en"
    target_lang = "ar"
    
    translated_text = gpt_translate(text, source_lang, target_lang)
    print(f"Translated text: {translated_text}")