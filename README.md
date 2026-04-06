# 🧠 ACIE-HADO  
### Adaptive Cognitive Intelligence Environment for Human-AI Decision Optimization

---

##  The Real Problem (What Nobody is Solving Properly)

In today’s world, humans are not failing because they lack intelligence —  
they are failing because they are **mentally overloaded**.

Students, developers, and managers are constantly forced to:
- make too many decisions  
- switch between tasks rapidly  
- handle emotional and stressful situations  
- work under cognitive fatigue  

This leads to:
- poor decisions  
- emotional mistakes (e.g., angry emails)  
- burnout  
- long-term regret  

---

##  Why Current AI Systems Fail

Modern AI assistants are built for **speed and productivity**, not for humans.

They:
- push tasks aggressively  
- respond instantly without context  
- ignore fatigue and stress  
- assume humans have unlimited mental capacity  

 Result:  
AI makes humans **faster**, but not **better decision-makers**.

---

##  Our Core Idea

**ACIE-HADO redefines AI as a cognitive protector, not just a task executor.**

Instead of asking:
> "What task should we do next?"

We ask:
> "Is the human in the right state to make this decision?"

---

##  Key Insight

> The biggest problem under pressure is not delay —  
> it is making a decision you regret later.

---

##  What Makes ACIE-HADO Unique

ACIE-HADO is not just another AI system.  
It is a **cognitive-aware decision environment** where the agent actively manages human mental capacity.

---

##  Signature Innovation: Decision Regret Forecasting

Before taking any action, the agent predicts:

> "Will this decision cause regret later?"

This allows the system to:
- delay risky decisions  
- prevent emotional mistakes  
- avoid poor judgment under stress  

 This shifts AI from **reactive → future-aware**

---

##  Other Key Innovations

###  Cognitive Fragmentation Index
Tracks how broken the user’s attention is due to:
- context switching  
- interruptions  
- task overload  

---

###  Emotional Spillover Modeling
Stress from one task affects future decisions.

Example:
> Angry email → affects next 2–3 decisions

---

###  Decision Debt System
Delayed decisions accumulate hidden cost:
- increasing stress  
- degrading performance  

---

###  Fair Team Load Redistribution
Delegation considers:
- teammate capacity  
- fairness  
- burnout risk  
- task suitability  

---

###  Recovery-Aware AI Behavior
The system actively:
- detects overload  
- triggers recovery  
- protects decision quality  

---

##  How the Agent Thinks

The agent follows a **human-like reasoning pipeline**:

1. Forecast future regret (`forecast_regret`)
2. Analyze mental state (`calculate_cognitive_score`)
3. Predict recovery needs (`predict_recovery`)
4. Protect from stress (`trigger_recovery_mode`)
5. Isolate high-risk tasks (`isolate_stressful_task`)
6. Automate low-stakes work (`activate_autopilot`)
7. Delegate intelligently (`redistribute_team_load`)
8. Stabilize and finish (`final_answer`)

 The goal is not speed  
 The goal is **safe, high-quality decisions**

---

##  Task Design

###  Easy — Student Fatigue Management
- Learn basic cognitive awareness  
- Automate low-stakes decisions  

---

###  Medium — Workload + Stress Management
- Handle emotional spillover  
- Reorder tasks intelligently  
- Protect user from overload  

---

###  Hard — High-Stakes Decision Protection
- User is critically overwhelmed  
- Agent must:
  - isolate emotional risks  
  - delegate workload  
  - prevent regret-heavy decisions  
  - maintain cognitive stability  

---

##  Reward Philosophy

This environment does NOT reward just task completion.

It rewards:
- correct timing of decisions  
- protecting users under stress  
- reducing future regret  
- intelligent delegation  

It penalizes:
- repeated actions  
- forcing decisions under fatigue  
- bad delegation  
- ignoring emotional context  

---

##  Example Behavior (Real Output)

```text
forecast_regret → predict_recovery → isolate_stressful_task → 
activate_autopilot → redistribute_team_load → final_answer
```

## How to Run

### Run Locally
To run the project completely locally utilizing our provided Docker framework:
1. Clone the repository.
2. Build the image: `docker build -t acie-hado .`
3. Run the evaluation setup: `docker run -it acie-hado python inference.py`

### Run Inference
Ensure your active environment possesses the necessary variables. Our integration prefers OpenAI compatible REST APIs (including HuggingFace wrappers).

**In Powershell:**
```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
$env:HF_TOKEN="your-hf-token"
python inference.py > inference_output.txt 2>&1
Get-Content inference_output.txt
```

### Expected Output Format
The resulting output log will align precisely with Hackathon/validation environments, structured accurately with brackets and boolean standards:

```text
[START] task=easy env=acie_hado model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action={"action_type":"calculate_cognitive_score"} reward=0.20 done=false error=null
[STEP] step=2 action={"action_type":"activate_autopilot","target_task_id":"lunch_order"} reward=0.50 done=false error=null
[STEP] step=3 action={"action_type":"final_answer"} reward=1.00 done=true error=null
[END] success=true steps=3 rewards=0.20,0.50,1.00
[SUMMARY] average_score=0.85
```

## Limitations and Future Work
- **Static Team Context:** Currently, team scores are provided statically per task state. Future work intends to construct parallel AI agents behaving dynamically as teammates.
- **Continuous Execution Length:** Bounded strictly to 10 max iterations. We look to expand this bounds as task difficulties scale up into day-long workflow architectures.
- **Eventual Front-End Dashboard:** Extending the application with an aesthetic React front-end charting real-time interactive spillover graphics to better visually monitor cognitive sustainability.
