from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
from src.model.ner import initialize_ner_model, process_text

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

ner_pipeline = initialize_ner_model()

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ner")
async def process_ner(request: Request):
    data = await request.json()
    text = data.get("text", "")
    converted_results = process_text(text, ner_pipeline)
    return {"results": converted_results}

@app.get("/test_ner")
async def test_ner():
    # Use multiple diverse sample texts to cover various entity types
    sample_texts = [
        "我來自台灣，是一個民主國家。",
        "李小明在2023年12月25日上午10點前往台北101大樓。",
        "美國總統拜登與中國領導人習近平會面，討論國際關係。",
        "基督教和伊斯蘭教是世界主要宗教。"
    ]
    all_results = []
    for text in sample_texts:
        results = process_text(text, ner_pipeline)
        all_results.extend(results)
    entity_types = list(set(result["entity_group"] for result in all_results))
    return {"results": all_results, "entity_types": entity_types}
