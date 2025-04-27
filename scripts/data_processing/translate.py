import os
from dotenv import load_dotenv
import openai

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
    # Load environment variables from .env file
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
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

def print_span(article_number, start_offset, end_offset, lang="en", base_path="data/raw"):
    article_path = f"{base_path}/{lang}/train-articles-subtask-3/article{article_number}.txt"
    with open(article_path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")
    span = text[start_offset:end_offset]
    print(span)

import re
import os

def clean_existing_tokens(text):
    # Remove anything like <<S_23>> or <</S_23>>
    return re.sub(r"<<\/?S_\d+>>", "", text)

def wrap_annotated_spans(article_path, spans, output_folder=None, token_prefix="S"):
    with open(article_path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")

    # STEP 1: clean previous tokens
    text = clean_existing_tokens(text)
    
    text_length = len(text)
    # Check if spans are within the text length
    valid_spans = []
    for start, end in spans:
        if 0 <= start < end <= text_length:
            valid_spans.append((start, end))
        else:
            print(f"⚠️ Skipping invalid span ({start}, {end}) for article {article_path}")

    valid_spans = sorted(valid_spans, key=lambda x: x[0], reverse=True)

    for idx, (start, end) in enumerate(valid_spans, 1):
        open_token = f"<<{token_prefix}_{idx}>>"
        close_token = f"<</{token_prefix}_{idx}>>"
        text = text[:start] + open_token + text[start:end] + close_token + text[end:]

    return text

def wrap_spans_from_file(labels_file, articles_folder, output_folder, lang="en", token_prefix="S"):
    """
    Reads a span annotation file and wraps the spans in the corresponding article files.
    Args:
        labels_file (str): Path to the span annotation file (e.g., train-labels-subtask-3-spans.txt).
        articles_folder (str): Path to the folder containing article text files.
        output_folder (str): Path to the folder to write wrapped articles.
        lang (str): Language code (default: 'en').
        token_prefix (str): Prefix for the span tokens (default: 'S').
    """
    import os
    from collections import defaultdict

    # Collect spans per article
    article_spans = defaultdict(list)
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            article_id, label, start, end = parts[:4]
            article_spans[article_id].append((int(start), int(end)))

    os.makedirs(output_folder, exist_ok=True)

    # For each article, wrap spans and write output
    for article_id, spans in article_spans.items():
        article_path = os.path.join(articles_folder, f"article{article_id}.txt")
        if not os.path.exists(article_path):
            print(f"Warning: Article file not found: {article_path}")
            continue
        wrapped_text = wrap_annotated_spans(article_path, spans, token_prefix=token_prefix)
        output_path = os.path.join(output_folder, f"article{article_id}.txt")
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(wrapped_text)