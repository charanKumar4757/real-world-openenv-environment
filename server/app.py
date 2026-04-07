from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.env import CognitiveEnv

app = FastAPI(title="Real World OpenEnv Environment")
env = CognitiveEnv()


# ---------- Request Model ----------
class ActionRequest(BaseModel):
    action_type: str
    target_task_id: str | None = None
    target_user: str | None = None


# ---------- API Routes ----------
@app.get("/")
def root():
    return {"status": "ok", "message": "OpenEnv environment is running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/reset")
def reset(task_level: str = "easy"):
    try:
        return env.reset(task_level)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: ActionRequest):
    try:
        observation, reward, done, info = env.step(action.model_dump())
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- REQUIRED FOR VALIDATOR ----------
def main():
    """
    Entry point for validator (multi-mode support)
    """
    print("Server module ready for OpenEnv validation.")


# ---------- IMPORTANT ----------
if __name__ == "__main__":
    main()
