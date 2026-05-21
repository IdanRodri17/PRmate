from fastapi import FastAPI, Request

app = FastAPI(title="PRmate")


@app.get("/")
def home():
    return {"app": "PRmate", "status": "alive"}


@app.post("/webhook")
async def webhook(request: Request):
    # Placeholder — real handler comes in V1
    payload = await request.json()
    return {"received": True, "event": payload.get("action", "unknown")}
