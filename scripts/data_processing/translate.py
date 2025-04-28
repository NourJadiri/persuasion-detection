import os
from dotenv import load_dotenv
import openai
import aiofiles

PERSUASION_TECHNIQUES = """
Name_Calling-Labeling
Guilt_by_Association
Doubt
Appeal_to_Hypocrisy
Questioning_the_Reputation
Flag_Waving
Appeal_to_Authority
Appeal_to_Popularity
Appeal_to_Values
Appeal_to_Fear-Prejudice
Straw_Man
Red_Herring
Whataboutism
Appeal_to_Pity
Causal_Oversimplification
False_Dilemma-No_Choice
Consequential_Oversimplification
False_Equivalence
Slogans
Conversation_Killer
Appeal_to_Time
Loaded_Language
Obfuscation-Vagueness-Confusion
Exaggeration-Minimisation
Repetition
""".strip().replace("\n", ", ")

def build_prompt(text: str, src: str, tgt: str) -> str:
    return (
        "You are a professional translator.\n\n"
        "Task:\n"
        f"1. Translate the following {src} text into natural, idiomatic {tgt}. Aim for a **neutral** register.\n"
        "2. Keep every span token exactly as it appears, including the text wrapped inside:\n"
        "   <<S_1>> … <</S_1>>, <<S_2>> … <</S_2>>, etc.\n"
        "3. Preserve the rhetorical nuance of each persuasion technique so they remain recognisable.\n"
        f"   Techniques list: {PERSUASION_TECHNIQUES}.\n"
        "4. Do NOT add or remove tokens, change their numbering, summarise or omit content.\n\n"
        "Source text:\n```text\n"
        f"{text}\n```\n\n"
        "Output (translated text with identical tokens):"
    )
    
def gpt_translate(text, source_lang, target_lang, client):
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
    # Check if the client is initialized
    if client is None:
        raise ValueError("Client is not initialized. Please provide a valid OpenAI client.")
    # Validate the input text
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    # Validate the source and target language codes
    if not isinstance(source_lang, str) or not source_lang.strip():
        raise ValueError("Source language code must be a non-empty string.")
    if not isinstance(target_lang, str) or not target_lang.strip():
        raise ValueError("Target language code must be a non-empty string.")

    # Construct the prompt for translation using language codes
    prompt = build_prompt(
        text=text,
        src=source_lang,
        tgt=target_lang
    )

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

async def async_gpt_translate(text, source_lang, target_lang, client):
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
    # Check if the client is initialized
    if client is None:
        raise ValueError("Client is not initialized. Please provide a valid OpenAI client.")
    # Validate the input text
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    # Validate the source and target language codes
    if not isinstance(source_lang, str) or not source_lang.strip():
        raise ValueError("Source language code must be a non-empty string.")
    if not isinstance(target_lang, str) or not target_lang.strip():
        raise ValueError("Target language code must be a non-empty string.")

    # Construct the prompt for translation using language codes
    prompt = build_prompt(
        text=text,
        src=source_lang,
        tgt=target_lang
    )

    # Call the OpenAI API for translation
    response = await client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "user", "content": prompt}
        ],
    )

    # Extract the translated text from the response
    translated_text = response.output_text

    return translated_text

def translate_file_to_language(input_path, target_lang, client, base_dir="data/processed"):
    """
    Translates the content of a file to the target language and saves it in the corresponding folder.

    Args:
        input_path (str): Path to the source file (e.g., data/processed/fr/article2318.txt).
        target_lang (str): Target language code (e.g., 'ru').
        client: Initialized OpenAI client.
        base_dir (str): Base directory for processed data.

    Returns:
        str: Path to the saved translated file.
    """
    # Extract source language and filename
    parts = input_path.split(os.sep)
    if len(parts) < 3:
        raise ValueError("Input path must be like data/processed/{src_lang}/filename.txt")
    src_lang = parts[2]
    filename = parts[-1]

    # Read the source text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Translate
    translated_text = gpt_translate(text, src_lang, target_lang, client)

    # Prepare output path
    output_dir = os.path.join(base_dir, target_lang, "wrapped-articles")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)

    # Save translated text using raw byte encoding
    with open(output_path, "wb") as f:
        f.write(translated_text.encode("utf-8"))

    return output_path


async def async_translate_file_to_language(input_path, target_lang, client, base_dir="data/processed"):
    # Extract source language and filename
    parts = input_path.split(os.sep)
    if len(parts) < 3:
        raise ValueError("Input path must be like data/processed/{src_lang}/filename.txt")
    src_lang = parts[2]
    filename = parts[-1]

    # Read the source text
    async with aiofiles.open(input_path, "r", encoding="utf-8") as f:
        text = await f.read()
    # Translate
    translated_text = await async_gpt_translate(text, src_lang, target_lang, client)
    # Prepare output path
    output_dir = os.path.join(base_dir, target_lang)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    # Save translated text using raw byte encoding
    async with aiofiles.open(output_path, "wb") as f:
        await f.write(translated_text.encode("utf-8"))
    return output_path


def count_characters_in_wrapped_articles(language_code, base_dir="data/processed"):
    """
    Counts the total number of characters in all text files inside
    data/processed/{language_code}/wrapped-articles.

    Args:
        language_code (str): The language code (e.g., 'en', 'fr').
        base_dir (str): The base directory path.

    Returns:
        int: Total number of characters in all text files.
    """
    folder = os.path.join(base_dir, language_code, "wrapped-articles")
    total_chars = 0
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                total_chars += len(f.read())
    return total_chars

def total_characters_all_languages(base_dir="data/processed"):
    """
    Computes the total number of characters across all language folders
    within the base directory.

    Args:
        base_dir (str): The base directory path containing language folders.

    Returns:
        int: Total number of characters across all language folders.
    """
    total_chars = 0
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found.")
        return 0

    for lang_code in os.listdir(base_dir):
        lang_path = os.path.join(base_dir, lang_code)
        # Check if it's a directory and if the 'wrapped-articles' subfolder exists
        articles_path = os.path.join(lang_path, "wrapped-articles")
        if os.path.isdir(lang_path) and os.path.isdir(articles_path):
            try:
                total_chars += count_characters_in_wrapped_articles(lang_code, base_dir)
            except FileNotFoundError:
                # Handle cases where wrapped-articles might exist but is empty or has issues
                print(f"Warning: Could not process folder for language '{lang_code}'. Skipping.")
            except Exception as e:
                print(f"An error occurred processing language '{lang_code}': {e}")

    return total_chars

# Example usage:
if __name__ == "__main__":
    en_count = count_characters_in_wrapped_articles('en')
    fr_count = count_characters_in_wrapped_articles('fr')
    print(f"Total characters in 'en': {en_count}")
    print(f"Total characters in 'fr': {fr_count}")
