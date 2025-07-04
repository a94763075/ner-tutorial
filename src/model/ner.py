from transformers import pipeline

def initialize_ner_model():
    model_name = "ckiplab/bert-base-chinese-ner"
    return pipeline("ner", model=model_name, grouped_entities=True)

def convert_to_serializable(data):
    if isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, (float, type(None))) or str(type(data)).find('float32') != -1:
        return float(data) if data is not None else None
    return data

def process_text(text, ner_pipeline):
    results = ner_pipeline(text)
    return convert_to_serializable(results)
