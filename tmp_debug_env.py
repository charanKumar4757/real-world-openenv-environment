from app.env import CognitiveEnv

env = CognitiveEnv()
state = env.reset('hard')
actions = [
    {'action_type': 'forecast_regret', 'target_task_id': None, 'target_user': None},
    {'action_type': 'predict_recovery', 'target_task_id': None, 'target_user': None},
    {'action_type': 'trigger_recovery_mode', 'target_task_id': None, 'target_user': None},
    {'action_type': 'isolate_stressful_task', 'target_task_id': 'angry_client_email', 'target_user': None},
    {'action_type': 'activate_autopilot', 'target_task_id': 'lunch_choice', 'target_user': None},
    {'action_type': 'redistribute_team_load', 'target_task_id': 'client_approval', 'target_user': 'Sara'},
]

for i, a in enumerate(actions, start=1):
    state, reward, done, info = env.step(a)
    print(f"after {i} {a['action_type']} reward {reward} done {done}")
    print('pending', [t['task_id'] for t in state.get('pending_tasks', [])])
    print('decision_debt', state.get('decision_debt'), 'cog', state.get('cognitive_score'))
    print('---')
