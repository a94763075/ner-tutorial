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
    test_text = "我來自台灣，是一個民主國家。"
    converted_results = process_text(test_text, ner_pipeline)
    return {"results": converted_results}
