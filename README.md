---
title: ACIE-HADO Environment
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
short_description: Adaptive Cognitive Intelligence Environment for Human-AI Decision Optimization
---

# ACIE-HADO: Adaptive Cognitive Intelligence Environment for Human-AI Decision Optimization

## Problem

Humans face **cognitive overload and decision fatigue** in daily life:
- Too many decisions → cognitive capacity exhausted
- Fatigue → poor decision quality
- Emotional spillover → wrong choices
- Context switching → fragmented attention
- Task overload → missed deadlines

## Solution

**ACIE-HADO** is an AI agent that learns to:
✅ Detect cognitive overload and fatigue  
✅ Predict recovery time  
✅ Activate "autopilot" for low-stakes decisions  
✅ Reorder tasks by urgency and cognitive cost  
✅ Distribute work to healthier team members  
✅ Protect emotional state with strategic delays  
✅ Track decision debt (decisions deferred to later review)  

## Project Identity

**Title**: Adaptive Cognitive Intelligence Environment for Human-AI Decision Optimization  
**Short Name**: ACIE-HADO  
**Acronym**: ACIE (Adaptive Cognitive Intelligence Environment), HADO (Human-AI Decision Optimization)

## Observation Space

The environment provides these observations to the agent:

```
cognitive_score: int (0-100)           Current cognitive capacity
fatigue_level: str                      low | medium | high | overwhelmed
decision_count: int                     Total decisions made so far
emotional_state: str                    neutral | stressed | overwhelmed
recovery_prediction: int                Minutes until cognitive recovery
decision_debt: int                      Deferred decisions pending review
spillover_level: int (0-100)            Emotional stress from recent tasks
pending_tasks: list                     Tasks waiting to be handled
team_scores: dict                       Cognitive scores of team members
autopilot_active: bool                  Is AI handling low-stakes tasks?
human_trust_score: int (0-100)          User trust in AI system
```

## Action Space

The agent can take these actions:

```
calculate_cognitive_score       Estimate user's current cognitive capacity
predict_recovery                Forecast when user can handle complex decisions
activate_autopilot              AI handles low-stakes decisions automatically
execute_task                    User executes a specific task
reorder_tasks                   Rearrange task queue by urgency & effort
delay_task                      Defer a task to later
trigger_recovery_mode           Short protected rest period
redistribute_team_load          Assign task to healthier team member
review_debt                     Review deferred decisions with user
isolate_stressful_task          Separate emotionally challenging task
final_answer                    Task complete, episode ends
forecast_regret                 Predict regret cost of decisions
provide_transparency            Explain AI's reasoning to user
reserve_recovery_window         Block time for user recovery
```

## Task Definitions

### Easy Task: Student Fatigue Scenario
- **Initial State**: Cognitive score 45, fatigue medium, 20 decisions made
- **Challenge**: User is tired but must choose an important elective course
- **Expected Actions**: Detect fatigue → Activate autopilot for lunch decision → Finish
- **Success**: Score ≥ 0.40

### Medium Task: Work Queue Optimization  
- **Initial State**: Cognitive score 30 (high fatigue), emotional stress, 45 decisions
- **Challenge**: Multiple tasks with emotional spillover from recent difficult email
- **Expected Actions**: Recognize stress → Trigger recovery → Reorder tasks → Automate low-stakes → Finish
- **Success**: Score ≥ 0.45

### Hard Task: Team Routing Under Crisis
- **Initial State**: Cognitive score 18 (overwhelming), 40 minutes to client presentation
- **Challenge**: Must protect user cognition while maintaining trust and output quality
- **Expected Actions**: Predict recovery → Isolate stressor → Activate autopilot → Route to Sara (score 88) → Finish
- **Success**: Score ≥ 0.50

## Reward Logic

### Positive Rewards
- `+0.2` Correct cognitive score estimation  
- `+0.5` Appropriate autopilot usage when needed  
- `+0.6` Successfully reducing cognitive overload  
- `+0.5` Smart task reordering (urgency + cost)  
- `+0.6` Successful team work distribution  
- `+0.5` Protecting user recovery needs  

### Negative Rewards
- `-0.5` Unnecessary overload created  
- `-0.6` Ignoring high fatigue/stress signals  
- `-0.3` Excessive decision debt accumulation  
- `-0.5` Repeated useless actions  
- `-0.4` Failed team member assignment  

## Running Locally

### 1. Start the Demo UI
```bash
python app.py
```
Then open http://localhost:7860

### 2. Test Inference (Fast Mode)
```powershell
$env:INFERENCE_MOCK="1"
$env:API_BASE_URL="https://example.com/v1"
$env:API_KEY="test-key"
$env:MODEL_NAME="gpt-4o-mini"
python inference.py
```

### 3. Test Inference (Real Mode)
```powershell
$env:API_BASE_URL="https://your-api/v1"
$env:API_KEY="your-real-key"
$env:MODEL_NAME="gpt-4o-mini"
python inference.py
```

## Docker Deployment

```bash
# Build
docker build -t acie-hado .

# Run
docker run -p 7860:7860 acie-hado
```

## Expected Inference Output

The inference script prints structured output that the validator parses:

```
[START] task=easy env=acie_hado model=gpt-4o-mini
[STEP] step=1 action={"action_type":"calculate_cognitive_score"} reward=0.2 done=false error=null
[STEP] step=2 action={"action_type":"activate_autopilot"} reward=0.5 done=false error=null
[END] task=easy success=true steps=2 score=0.7
```

## Architecture

```
real-world-openenv-environment/
├── app.py                      # Gradio UI
├── app/
│   ├── env.py                  # Core environment (reset/state/step)
│   ├── models.py               # Data models (Observation, Action, Task)
│   ├── tasks.py                # Task definitions (easy/medium/hard)
│   ├── grader.py               # Scoring logic
│   ├── reward.py               # Reward calculation
├── inference.py                # Hackathon validator script
├── openenv.yaml                # OpenEnv specification
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container config
└── README.md                   # This file
```

## Features

### Core (Phase 1)
- ✅ Cognitive budget engine (score 0-100)
- ✅ Recovery time prediction
- ✅ Autopilot mode for low-stakes tasks
- ✅ Task reordering by urgency + cognitive cost

### Advanced (Phase 2)
- ✅ Decision debt tracking
- ✅ Emotional spillover detection & shielding
- ✅ Team member cognitive scoring
- ✅ Regret forecasting

### Future (Phase 3)
- Pattern memory (detect fatigue cycles)
- Stress testing & resilience scoring
- Context switching penalty detection
- Long-term user profiling

## Evaluation Criteria

### Grading (0.0 - 1.0)
- **Correct sequence**: Did agent follow expected strategy?
- **State protection**: Did cognitive score stay above threshold?
- **Debt management**: Did decision debt stay low?
- **Trust preservation**: Did human_trust_score stay above 40?
- **Efficiency**: Did agent finish in reasonable steps?

## Hackathon Submission Notes

This project is part of the **Meta PyTorch Hackathon x Scaler School of Technology**.

**Phase 2 Requirements:**
- ✅ inference.py with structured stdout ([START]/[STEP]/[END])
- ✅ Proper environment variable handling (API_BASE_URL, API_KEY, MODEL_NAME)
- ✅ Graceful error handling without crashes
- ✅ Working Docker build

**Phase 3 Requirements:**
- Logic validation of environment mechanics
- Task completion within step limits
- Proper reward calculations

## Running Tests

```bash
# Quick mock test
$env:INFERENCE_MOCK="1"; python inference.py

# Full environment test
python -c "from app.env import CognitiveEnv; env = CognitiveEnv(); print(env.reset())"

# Syntax validation
python -m py_compile inference.py app/env.py app/models.py
```

## Dependencies

- Python 3.10+
- `gradio` - Web UI
- `openai` - LLM client
- `pydantic` - Data validation
- `fastapi` (optional)
- `uvicorn` (optional)

Check [requirements.txt](requirements.txt) for pinned versions.

## Key Insights

1. **Cognitive score is the resource**: Everything else derives from protecting this
2. **Emotional state matters**: Spillover affects decision quality
3. **Team intelligence is asymmetric**: Route heavy tasks to healthier members
4. **Autopilot has a cost**: Creates decision debt that needs review
5. **Recovery time is precious**: Protect it strategically before high-stakes events

## References

- [OpenEnv Spec](https://github.com/openenv-ai/openenv)
- [Cognitive Load Theory](https://en.wikipedia.org/wiki/Cognitive_load)
- [Decision Fatigue Research](https://en.wikipedia.org/wiki/Decision_fatigue)

---

**Built for**: Meta PyTorch Hackathon x Scaler School of Technology  
**Submission**: [GitHub](https://github.com/yourusername/real-world-openenv-environment)
