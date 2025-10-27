import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def test():
    return {"response": "天満だお"}

if __name__ == "__main__":
    uvicorn.run("TENMA:app", host="127.0.0.1", port=8000, reload=True)
