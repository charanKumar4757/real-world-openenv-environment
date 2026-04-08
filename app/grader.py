def grade_easy(info, final_state):
    score = 0.0
    pending = final_state.get("pending_tasks", [])
    cognitive_score = final_state.get("cognitive_score", 0)

    # Did agent handle all low-stakes tasks?
    low_stakes_remaining = [t for t in pending if t.get("is_low_stakes")]
    if len(low_stakes_remaining) == 0:
        score += 0.5

    # Did agent protect cognitive score?
    if cognitive_score >= 40:
        score += 0.3
    elif cognitive_score >= 30:
        score += 0.1

    # Did agent maintain trust?
    if final_state.get("human_trust_score", 0) >= 60:
        score += 0.2

    return min(1.0, round(score, 2))


def grade_medium(info, final_state):
    score = 0.0
    pending = final_state.get("pending_tasks", [])
    cognitive_score = final_state.get("cognitive_score", 0)

    # Did agent handle stressful tasks?
    stressful_remaining = [t for t in pending if t.get("is_stressful")]
    if len(stressful_remaining) == 0:
        score += 0.4

    # Did agent handle low-stakes tasks?
    low_stakes_remaining = [t for t in pending if t.get("is_low_stakes")]
    if len(low_stakes_remaining) < 2:
        score += 0.2

    # Did agent protect cognitive score?
    if cognitive_score >= 35:
        score += 0.3
    elif cognitive_score >= 20:
        score += 0.1

    # Did emotional state improve?
    if final_state.get("emotional_state") == "neutral":
        score += 0.1

    return min(1.0, round(score, 2))


def grade_hard(info, final_state):
    score = 0.0
    pending = final_state.get("pending_tasks", [])
    cognitive_score = final_state.get("cognitive_score", 0)

    critical_ids = ["angry_client_email", "client_approval", "lunch_choice"]
    critical_remaining = [t for t in pending if t.get("task_id") in critical_ids]

    # Did agent handle all critical tasks?
    if len(critical_remaining) == 0:
        score += 0.5

    # Did agent protect cognitive score?
    if cognitive_score >= 30:
        score += 0.3
    elif cognitive_score >= 20:
        score += 0.1

    # Did agent maintain trust?
    if final_state.get("human_trust_score", 0) >= 40:
        score += 0.2

    return min(1.0, round(score, 2))
