from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
from transformers import pipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model_name = "ckiplab/bert-base-chinese-ner"
ner_pipeline = pipeline("ner", model=model_name, grouped_entities=True)

def convert_to_serializable(data):
    if isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, (float, type(None))) or str(type(data)).find('float32') != -1:
        return float(data) if data is not None else None
    return data

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ner")
async def process_ner(request: Request):
    data = await request.json()
    text = data.get("text", "")
    results = ner_pipeline(text)
    converted_results = convert_to_serializable(results)
    return {"results": converted_results}
