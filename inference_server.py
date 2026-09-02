from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   
        "http://127.0.0.1:5173",   
        "http://localhost:3000",   
        "https://divineframing.com"  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


tokenizer = AutoTokenizer.from_pretrained("./my-sacred-model", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("./my-sacred-model", local_files_only=True)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    input_text = data.get("input", "")

    inputs = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=100,
        min_length=5,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output": decoded}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_server:app", host="127.0.0.1", port=8000, reload=True)
