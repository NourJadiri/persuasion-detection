def get_helsinki_nlp_translation_models():
    """
    Returns a list of Helsinki-NLP model names for translation between all language pairs
    relevant to the 5 Slavic languages (Bulgarian, Polish, Croatian, Slovene, Russian)
    and all other available languages in the dataset.
    """
    # Language codes in the dataset
    lang_codes = [
        'ar', 'bg', 'en', 'es', 'fr', 'ge', 'gr', 'it', 'ka', 'po', 'pt', 'ru', 'sl'
    ]
    # Map dataset codes to Helsinki-NLP codes
    helsinki_lang_map = {
        'ar': 'ar',
        'bg': 'bg',
        'en': 'en',
        'es': 'es',
        'fr': 'fr',
        'ge': 'hr',  # ge = Croatian (hr)
        'gr': 'el',  # gr = Greek (el)
        'it': 'it',
        'ka': 'ka',
        'po': 'pl',  # po = Polish (pl)
        'pt': 'pt',
        'ru': 'ru',
        'sl': 'sl',
    }
    # 5 Slavic target languages
    slavic_targets = ['bg', 'pl', 'hr', 'sl', 'ru']
    # Map dataset codes to slavic targets
    dataset_to_sla = {
        'bg': 'bg',
        'po': 'pl',
        'ge': 'hr',
        'sl': 'sl',
        'ru': 'ru',
    }
    # All source languages
    sources = [helsinki_lang_map[code] for code in lang_codes]
    # All target languages (Slavic only)
    targets = slavic_targets
    # Build model names, skip same source==target
    models = []
    for src in sources:
        for tgt in targets:
            if src != tgt:
                models.append(f"Helsinki-NLP/opus-mt-{src}-{tgt}")
    return models

if __name__ == "__main__":
    # Example usage
    models = get_helsinki_nlp_translation_models()
    print(models)
    print(f"Total models: {len(models)}")