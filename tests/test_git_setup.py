import subprocess
import os

def run_git(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    print(f"CMD: {cmd}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
    return result

def test_git_setup_and_commit():
    # 1. Check if .git exists, if not git init
    if not os.path.exists(".git"):
        res = run_git("git init")
        assert res.returncode == 0

    # Ensure git user config exists for commits in test environment
    run_git("git config user.name \"ForgeAI Agent\"")
    run_git("git config user.email \"agent@forgeai.local\"")

    # 2. Check git status
    status_res = run_git("git status")
    
    # 3. Create new branch 'forgeai/test-run'
    # Check current branch or switch/create branch
    branch_res = run_git("git checkout -b forgeai/test-run")
    if branch_res.returncode != 0:
        # If already exists or main/master
        run_git("git checkout forgeai/test-run || git checkout -b forgeai/test-run")

    # 4. Add all files
    add_res = run_git("git add .")
    assert add_res.returncode == 0

    # 5. Commit all current changes
    commit_res = run_git("git commit -m \"Initial commit: setup repository and add project files\"")
    # If nothing to commit (e.g., already committed), commit_res might have non-zero or zero, but let's check
    print(f"Commit return code: {commit_res.returncode}")

    # 6. Final git status check
    final_status = run_git("git status")
    assert "forgeai/test-run" in final_status.stdout or "On branch forgeai/test-run" in final_status.stdout
