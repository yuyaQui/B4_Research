from fastapi import FastAPI 
import uvicorn 




app = FastAPI()


@app.get("/")
def hello():
    return "Hello World"

@app.get("/yuya")
def yuya():
    print("ゆーやだお")
    return "Hello Yuya"

@app.post("/kikuchi")
def kikuchi(kikuchi_data):
    print("くっちーだお")
    print(kikuchi_data)
    return f"kikuchi dataは{kikuchi_data}"


if __name__ == "__main__":
    uvicorn.run("test_server:app", host="0.0.0.0", port=8000, reload=True)
 
