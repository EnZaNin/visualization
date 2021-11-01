from fastapi import FastAPI

app_provider = FastAPI()


@app_provider.get("/")
async def root():
    return {"message": "Hello World"}