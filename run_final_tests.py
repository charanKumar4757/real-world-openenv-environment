#!/usr/bin/env python3
"""
Final test runner and git sync script
Tests all components and prepares for deployment
"""
import subprocess
import os
import sys

os.chdir("c:\\Users\\bavug\\Downloads\\real-world OpenEnv environment")

print("=" * 70)
print("ACIE-HADO FINAL TEST & DEPLOYMENT SCRIPT")
print("=" * 70)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from src.env import CognitiveEnv
    from src.grader import grade_easy, grade_medium, grade_hard
    from src.models import Observation, Action
    from src.reward import calculate_reward
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Environment functionality
print("\n[2/5] Testing environment...")
try:
    env = CognitiveEnv()
    for level in ['easy', 'medium', 'hard']:
        state = env.reset(level)
        assert 'cognitive_score' in state
        assert 'situation_summary' in state
        assert 'pending_tasks' in state
        print(f"  ✅ {level:6s} task loads correctly (score={state['cognitive_score']})")
except Exception as e:
    print(f"❌ Environment test failed: {e}")
    sys.exit(1)

# Test 3: Grader functionality
print("\n[3/5] Testing graders...")
try:
    from src.grader import clamp_score
    env = CognitiveEnv()
    for level, grader in [('easy', grade_easy), ('medium', grade_medium), ('hard', grade_hard)]:
        env.reset(level)
        env.step({"action_type": "calculate_cognitive_score", "target_task_id": None, "target_user": None})
        score = grader(env.__dict__, env.state())
        assert 0 < score < 1, f"Score out of range: {score}"
        print(f"  ✅ {level:6s} grader works (score={score:.2f})")
except Exception as e:
    print(f"❌ Grader test failed: {e}")
    sys.exit(1)

# Test 4: Inference mock mode
print("\n[4/5] Testing inference (mock mode)...")
try:
    os.environ['INFERENCE_MOCK'] = '1'
    result = subprocess.run(
        [sys.executable, 'inference.py'],
        capture_output=True,
        text=True,
        timeout=30
    )
    output = result.stdout
    
    # Check for expected output format
    assert '[START]' in output, "Missing [START] marker"
    assert '[STEP]' in output, "Missing [STEP] marker"
    assert '[END]' in output, "Missing [END] marker"
    
    # Count tasks completed
    start_count = output.count('[START]')
    end_count = output.count('[END]')
    print(f"  ✅ Inference mock: {start_count} tasks started, {end_count} tasks completed")
    if 'success=true' in output:
        print(f"  ✅ Mock tasks passed validation")
except subprocess.TimeoutExpired:
    print("❌ Inference test timed out")
    sys.exit(1)
except AssertionError as e:
    print(f"❌ Inference test failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Inference test error: {e}")
    sys.exit(1)

# Test 5: Git status
print("\n[5/5] Git repository status...")
try:
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    changes = result.stdout.strip()
    
    if changes:
        print(f"  📝 Uncommitted changes detected:")
        change_lines = changes.split('\n')
        for line in change_lines[:10]:
            print(f"     {line}")
        if len(change_lines) > 10:
            extra = len(change_lines) - 10
            print(f"     ... and {extra} more files")
    else:
        print("  ✅ No uncommitted changes")
    
    # Get remote info
    result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True)
    remotes = result.stdout.strip()
    if remotes:
        print(f"\n  📍 Git remotes configured:")
        for line in remotes.split('\n'):
            if line:
                print(f"     {line}")
    else:
        print("  ⚠️  No remotes configured. Run: git remote add origin <url>")
        
except Exception as e:
    print(f"⚠️  Git check failed: {e}")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - READY FOR DEPLOYMENT")
print("=" * 70)
print("\nNext steps:")
print("  1. Commit changes: git add -A && git commit -m 'Phase 3 fixes complete'")
print("  2. Push to GitHub:  git push origin main")
print("  3. Push to HF Spaces: git push spaces main  (if configured)")
print("\n" + "=" * 70)
