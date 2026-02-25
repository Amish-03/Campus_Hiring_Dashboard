"""
SSH Refinement Runner — Orchestrates remote execution.

Connects to the remote CUDA PC via SSH, locates the project,
pulls latest code, and runs the refinement pipeline.

Configuration via environment variables:
    SSH_HOST     — Remote PC IP/hostname
    SSH_USER     — SSH username
    SSH_PASS     — SSH password
    SSH_TIMEOUT  — Connection timeout (default: 15s)

Usage:
    set SSH_HOST=<your-host>
    set SSH_USER=<your-user>
    set SSH_PASS=<your-password>
    python ssh_refinement_runner.py
"""

import subprocess
import sys
import os
import time

# ── SSH Configuration (from environment) ──────────────────
SSH_HOST = os.environ.get("SSH_HOST", "")
SSH_USER = os.environ.get("SSH_USER", "")
SSH_PASS = os.environ.get("SSH_PASS", "")
SSH_TIMEOUT = int(os.environ.get("SSH_TIMEOUT", "15"))

if not SSH_HOST or not SSH_USER or not SSH_PASS:
    print("ERROR: SSH_HOST, SSH_USER, and SSH_PASS environment variables must be set.")
    print("  Example:")
    print('    set SSH_HOST=192.168.1.100')
    print('    set SSH_USER=myuser')
    print('    set SSH_PASS=mypassword')
    print('    python ssh_refinement_runner.py')
    sys.exit(1)

# Possible project locations (Windows paths on remote)
PROJECT_SEARCH_PATHS = [
    rf"C:\Users\{SSH_USER}\Desktop",
    rf"C:\Users\{SSH_USER}\Documents",
    rf"C:\Users\{SSH_USER}",
    rf"C:\Users\{SSH_USER}\Downloads",
    r"D:\\",
]

PROJECT_MARKERS = ["requirements.txt", "src\\main.py"]
DB_RELATIVE_PATH = r"data\campus_hiring.db"


def ssh_command(cmd: str, timeout: int = SSH_TIMEOUT) -> tuple:
    """Execute a command via SSH using sshpass. Returns (stdout, stderr, returncode)."""
    full_cmd = [
        "sshpass", "-p", SSH_PASS,
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={timeout}",
        f"{SSH_USER}@{SSH_HOST}",
        cmd
    ]

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except FileNotFoundError:
        return _ssh_with_plink(cmd, timeout)
    except subprocess.TimeoutExpired:
        return "", "Timeout expired", 1


def _ssh_with_plink(cmd: str, timeout: int) -> tuple:
    """Fallback: Use plink (PuTTY) or plain ssh."""
    try:
        result = subprocess.run(
            ["plink", "-ssh", f"{SSH_USER}@{SSH_HOST}", "-pw", SSH_PASS, "-batch", cmd],
            capture_output=True, text=True, timeout=timeout + 5,
            encoding="utf-8", errors="replace",
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-o", f"ConnectTimeout={timeout}",
             f"{SSH_USER}@{SSH_HOST}", cmd],
            capture_output=True, text=True, timeout=timeout + 10,
            encoding="utf-8", errors="replace",
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1


def ssh_run(cmd: str, description: str = "", timeout: int = SSH_TIMEOUT, retries: int = 1) -> str:
    """Run SSH command with retry logic and logging."""
    if description:
        print(f"  → {description}")

    for attempt in range(retries + 1):
        stdout, stderr, rc = ssh_command(cmd, timeout=timeout)
        if rc == 0:
            if stdout:
                for line in stdout.split("\n"):
                    print(f"    {line}")
            return stdout
        else:
            if attempt < retries:
                print(f"    ⚠ Failed (attempt {attempt+1}), retrying...")
                time.sleep(2)
            else:
                print(f"    ✗ FAILED: {stderr[:200]}")
                return ""
    return ""


def locate_project() -> str:
    """Search for the project directory on the remote machine."""
    print("\n" + "=" * 60)
    print("  STEP 1: Locating Project on Remote PC")
    print("=" * 60)

    for base_path in PROJECT_SEARCH_PATHS:
        for dirname in ["Campus_Hiring_Dashboard", "campus_hiring_nlp"]:
            test_path = f"{base_path}\\{dirname}"
            check_cmd = f'if exist "{test_path}\\requirements.txt" echo FOUND:{test_path}'
            result = ssh_run(check_cmd, f"Checking {test_path}...")
            if "FOUND" in result:
                print(f"  ✓ Project found: {test_path}")
                return test_path

    # Broader search
    print("  Searching with dir /s...")
    result = ssh_run(
        f'dir /s /b "C:\\Users\\{SSH_USER}\\requirements.txt" 2>nul',
        "Searching for requirements.txt...",
        timeout=30
    )
    if result:
        for line in result.split("\n"):
            line = line.strip()
            if "requirements.txt" in line:
                project_dir = line.replace("\\requirements.txt", "")
                check = ssh_run(f'if exist "{project_dir}\\src\\main.py" echo FOUND')
                if "FOUND" in check:
                    print(f"  ✓ Project found: {project_dir}")
                    return project_dir

    print("  ✗ Could not locate project automatically!")
    return ""


def main():
    print("\n" + "=" * 60)
    print("  SSH REFINEMENT RUNNER")
    print(f"  Target: {SSH_USER}@{SSH_HOST}")
    print("=" * 60)

    # Step 0: Test SSH connection
    print("\n  Testing SSH connection...")
    result = ssh_run("echo CONNECTION_OK", "Connecting...", retries=1)
    if "CONNECTION_OK" not in result:
        print("  ✗ SSH connection failed. Please check credentials and network.")
        sys.exit(1)
    print("  ✓ SSH connection successful")

    # Step 1: Locate project
    project_path = locate_project()
    if not project_path:
        print("  ✗ Cannot proceed without project path. Exiting.")
        sys.exit(1)

    # Step 2: Verify CUDA
    print("\n" + "=" * 60)
    print("  STEP 2: Verifying Environment")
    print("=" * 60)

    ssh_run(
        f'cd /d "{project_path}" && python -c "import torch; print(f\'CUDA: {{torch.cuda.is_available()}}, GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}\')"',
        "Checking CUDA...",
        timeout=30
    )

    # Step 3: Pull latest code
    print("\n" + "=" * 60)
    print("  STEP 3: Pulling Latest Code")
    print("=" * 60)

    ssh_run(
        f'cd /d "{project_path}" && git pull origin main',
        "Pulling from GitHub...",
        timeout=30,
        retries=1
    )

    # Step 4: Verify DB exists
    print("\n" + "=" * 60)
    print("  STEP 4: Verifying Database")
    print("=" * 60)

    db_check = ssh_run(
        f'if exist "{project_path}\\{DB_RELATIVE_PATH}" echo DB_EXISTS',
        "Checking campus_hiring.db...",
    )
    if "DB_EXISTS" not in db_check:
        print("  ✗ Database not found!")
        sys.exit(1)
    print("  ✓ Database found")

    # Step 5: Run refinement
    print("\n" + "=" * 60)
    print("  STEP 5: Running Refinement Pipeline (this takes ~5 min)")
    print("=" * 60)

    ssh_run(
        f'cd /d "{project_path}" && python -m src.refinement.remote_db_refinement',
        "Executing refinement...",
        timeout=600,
        retries=1
    )

    # Step 6: Verify results
    print("\n" + "=" * 60)
    print("  STEP 6: Verifying Results")
    print("=" * 60)

    ssh_run(
        f'cd /d "{project_path}" && python -c "'
        f"import sqlite3; conn=sqlite3.connect('data/campus_hiring.db'); c=conn.cursor(); "
        f"c.execute('SELECT COUNT(1) FROM company_normalization_map'); print(f'Normalization map: {{c.fetchone()[0]}} entries'); "
        f"c.execute('SELECT COUNT(1) FROM refined_drives'); print(f'Refined drives: {{c.fetchone()[0]}} drives'); "
        f"conn.close()"
        f'"',
        "Checking new tables...",
        timeout=15
    )

    print("\n" + "=" * 60)
    print("  COMPLETE ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
