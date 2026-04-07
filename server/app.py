from fastapi import FastAPI
from pydantic import BaseModel
from app.env import CognitiveEnv

app = FastAPI()

env = CognitiveEnv()
current_state = None

class StepInput(BaseModel):
    action: dict

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/reset")
def reset(task_level: str = "easy"):
    global current_state
    current_state = env.reset(task_level)
    return {"message": "environment reset", "state": current_state}

@app.post("/step")
def step(data: StepInput):
    global current_state
    if current_state is None:
        return {"error": "Environment not reset"}
    next_state, reward, done, info = env.step(data.action)
    current_state = next_state
    return {
        "reward": reward,
        "done": done,
        "state": current_state,
        "info": info
    }

@app.get("/state")
def get_state():
    if current_state is None:
        return {"error": "Environment not reset"}
    return current_state

import uvicorn

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
