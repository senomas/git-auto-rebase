import subprocess
import google.generativeai as genai
import os
import argparse
import sys
import datetime
import re
import logging
import tempfile
import json  # Used to pass data to editor script

# --- Configuration ---

# Configure logging
logging.basicConfig(level=logging.WARN, format="%(levelname)s: %(message)s")

# Attempt to get API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set.")
    logging.error(
        "Please obtain an API key from Google AI Studio (https://aistudio.google.com/app/apikey)"
    )
    logging.error("and set it as an environment variable.")
    sys.exit(1)

# Configure the Gemini AI Client
try:
    genai.configure(api_key=API_KEY)
    MODEL_NAME = os.getenv("GEMINI_MODEL")
    if not MODEL_NAME:
        logging.error("GEMINI_MODEL environment variable not set.")
        logging.error(
            "Please set the desired Gemini model name (e.g., 'gemini-1.5-flash-latest')."
        )
        logging.error("  export GEMINI_MODEL='gemini-1.5-flash-latest'  (Linux/macOS)")
        logging.error("  set GEMINI_MODEL=gemini-1.5-flash-latest       (Windows CMD)")
        logging.error(
            "  $env:GEMINI_MODEL='gemini-1.5-flash-latest'   (Windows PowerShell)"
        )
        sys.exit(1)
    model = genai.GenerativeModel(MODEL_NAME)
    logging.info(f"Using Gemini model: {MODEL_NAME}")
except Exception as e:
    logging.error(f"Error configuring Gemini AI: {e}")
    sys.exit(1)

# --- Git Helper Functions (Copied from previous script) ---


def run_git_command(command_list, check=True, capture_output=True, env=None):
    """
    Runs a Git command as a list of arguments and returns its stdout.
    Handles errors and returns None on failure if check=True.
    Allows passing environment variables.
    """
    full_command = []
    try:
        full_command = ["git"] + command_list
        logging.info(f"Running command: {' '.join(full_command)}")
        cmd_env = os.environ.copy()
        if env:
            cmd_env.update(env)
        result = subprocess.run(
            full_command,
            check=check,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=cmd_env,
        )
        if result.stdout:
            logging.info(f"Command output:\n{result.stdout[:200]}...")
            return result.stdout.strip() if capture_output else ""
        return ""
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing Git command: {' '.join(full_command)}")
        stderr_safe = (
            e.stderr.strip().encode("utf-8", "replace").decode("utf-8")
            if e.stderr
            else ""
        )
        stdout_safe = (
            e.stdout.strip().encode("utf-8", "replace").decode("utf-8")
            if e.stdout
            else ""
        )
        logging.error(f"Exit Code: {e.returncode}")
        if stderr_safe:
            logging.error(f"Stderr: {stderr_safe}")
        if stdout_safe:
            logging.error(f"Stdout: {stdout_safe}")
        return None
    except FileNotFoundError:
        logging.error(
            "Error: 'git' command not found. Is Git installed and in your PATH?"
        )
        sys.exit(1)
    except Exception as e:
        logging.error(f"Running command: {' '.join(full_command)}")
        logging.error(f"An unexpected error occurred running git: {e}")
        return None


def check_git_repository():
    output = run_git_command(["rev-parse", "--is-inside-work-tree"])
    return output == "true"


def get_current_branch():
    return run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])


def create_backup_branch(branch_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    backup_branch_name = f"{branch_name}-backup-{timestamp}"
    logging.info(
        f"Attempting to create backup branch: {backup_branch_name} from {branch_name}"
    )
    output = run_git_command(["branch", backup_branch_name, branch_name])
    if output is not None:
        logging.info(f"Successfully created backup branch: {backup_branch_name}")
        return backup_branch_name
    else:
        logging.error(f"Failed to create backup branch.")
        return None


def get_commit_range(upstream_ref, current_branch):
    logging.info(
        f"Finding merge base between '{upstream_ref}' and '{current_branch}'..."
    )
    merge_base = run_git_command(["merge-base", upstream_ref, current_branch])
    if not merge_base:
        logging.error(
            f"Could not find merge base between '{upstream_ref}' and '{current_branch}'."
        )
        return None, None
    logging.info(f"Found merge base: {merge_base}")
    commit_range = f"{merge_base}..{current_branch}"
    return commit_range, merge_base


def get_commits_in_range(commit_range):
    # Use --format=%H %s to get full hash for later matching if needed
    log_output = run_git_command(
        ["log", "--pretty=format:%h %H %s", "--reverse", commit_range]
    )
    if log_output is not None:
        commits = log_output.splitlines()
        # Store as list of dicts for easier access
        commit_data = []
        for line in commits:
            parts = line.split(" ", 2)
            if len(parts) == 3:
                commit_data.append(
                    {"short_hash": parts[0], "full_hash": parts[1], "subject": parts[2]}
                )
        logging.info(f"Found {len(commit_data)} commits in range {commit_range}.")
        return commit_data
    return []


def get_diff_in_range(commit_range):
    """Gets the combined diffstat and patch for the specified range."""
    diff_output = run_git_command(["diff", "--patch-with-stat", commit_range])
    if diff_output is not None:
        logging.info(
            f"Generated diff for range {commit_range} (length: {len(diff_output)} chars)."
        )
    else:
        logging.warning(f"Could not generate diff for range {commit_range}.")
    return diff_output if diff_output is not None else ""


# --- AI Interaction ---


def generate_reword_suggestion_prompt(commit_range, merge_base, commits_data, diff):
    """
    Creates a prompt asking the AI to identify commits needing rewording
    and to generate the full new commit message for each.
    """
    # Format commit list for the prompt using only short hash and subject
    commit_list_str = (
        "\n".join([f"- {c['short_hash']} {c['subject']}" for c in commits_data])
        if commits_data
        else "No commits in range."
    )

    prompt = f"""
You are an expert Git assistant specializing in commit message conventions. Your task is to analyze the provided Git commit history within the range `{commit_range}` and identify commits whose messages should be improved using `reword` during an interactive rebase (`git rebase -i {merge_base}`).

**Goal:** For each commit needing improvement, generate a **complete, new commit message** (subject and body) that adheres strictly to standard Git conventions.

**Git Commit Message Conventions to Adhere To:**

1.  **Subject Line:** Concise, imperative summary (max 50 chars). Capitalized. No trailing period. Use types like `feat:`, `fix:`, `refactor:`, `perf:`, `test:`, `build:`, `ci:`, `docs:`, `style:`, `chore:`. Example: `feat: Add user authentication endpoint`
2.  **Blank Line:** Single blank line between subject and body.
3.  **Body:** Explain 'what' and 'why' (motivation, approach, contrast with previous behavior). Wrap lines at 72 chars. Omit body ONLY for truly trivial changes where the subject is self-explanatory. Example:
    ```
    refactor: Improve database query performance

    The previous implementation used multiple sequential queries
    to fetch related data, leading to N+1 problems under load.

    This change refactors the data access layer to use a single
    JOIN query, significantly reducing database roundtrips and
    improving response time for the user profile page.
    ```

**Provided Context:**

1.  **Commit Range:** `{commit_range}`
2.  **Merge Base Hash:** `{merge_base}`
3.  **Commits in Range (Oldest First - Short Hash & Subject):**
```
{commit_list_str}
```
4.  **Combined Diff for the Range (`git diff --patch-with-stat {commit_range}`):**
```diff
{diff if diff else "No differences found or unable to get diff."}
```

**Instructions:**

1.  Analyze the commits listed above, focusing on their subjects and likely content based on the diff.
2.  Identify commits whose messages are unclear, too long, lack a type prefix, are poorly formatted, or don't adequately explain the change.
3.  For **each** commit you identify for rewording, output a block EXACTLY in the following format:
    ```text
    REWORD: <short_hash_to_reword>
    NEW_MESSAGE:
    <Generated Subject Line Adhering to Conventions>

    <Generated Body Line 1 Adhering to Conventions>
    <Generated Body Line 2 Adhering to Conventions>
    ...
    <Generated Body Last Line Adhering to Conventions>
    END_MESSAGE
    ```
    * Replace `<short_hash_to_reword>` with the short hash from the commit list.
    * Replace `<Generated Subject Line...>` with the new subject line you generate.
    * Replace `<Generated Body Line...>` with the lines of the new body you generate (if a body is needed). Ensure a blank line between subject and body, and wrap body lines at 72 characters. If no body is needed, omit the body lines but keep the blank line after the Subject.
    * The `END_MESSAGE` line marks the end of the message for one commit.
4.  Provide *only* blocks in the specified `REWORD:...END_MESSAGE` format. Do not include explanations, introductory text, or any other formatting. If no rewording is suggested, output nothing.

Now, analyze the provided context and generate the reword suggestions with complete new messages.
"""
    return prompt


def parse_reword_suggestions(ai_response_text, commits_data):
    """Parses AI response for REWORD:/NEW_MESSAGE:/END_MESSAGE blocks."""
    reword_plan = {}  # Use dict: {short_hash: new_message_string}
    commit_hashes = {c["short_hash"] for c in commits_data}  # Set of valid short hashes

    # Regex to find blocks
    pattern = re.compile(
        r"REWORD:\s*(\w+)\s*NEW_MESSAGE:\s*(.*?)\s*END_MESSAGE",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(ai_response_text)

    for match in matches:
        reword_hash = match[0].strip()
        new_message = match[1].strip()  # Includes Subject: and body

        if reword_hash in commit_hashes:
            reword_plan[reword_hash] = new_message
            logging.debug(
                f"Parsed reword suggestion for {reword_hash}:\n{new_message[:100]}..."
            )
        else:
            logging.warning(
                f"Ignoring invalid reword suggestion (hash {reword_hash} not in range)."
            )

    return reword_plan


# --- Automatic Rebase Logic ---


def create_rebase_sequence_editor_script(script_path, reword_plan):
    """Creates the python script for GIT_SEQUENCE_EDITOR (changes pick to reword)."""
    hashes_to_reword = set(reword_plan.keys())

    script_content = f"""#!/usr/bin/env python3
import sys
import logging
import re
import os

logging.basicConfig(level=logging.WARN, format="%(levelname)s: %(message)s")

todo_file_path = sys.argv[1]
logging.info(f"GIT_SEQUENCE_EDITOR script started for: {{todo_file_path}}")

hashes_to_reword = {hashes_to_reword!r}
logging.info(f"Applying rewording for hashes: {{hashes_to_reword}}")

new_lines = []
try:
    with open(todo_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            new_lines.append(line)
            continue

        match = re.match(r"^(\w+)\s+([0-9a-fA-F]+)(.*)", stripped_line)
        if match:
            action = match.group(1).lower()
            commit_hash = match.group(2)
            rest_of_line = match.group(3)

            if commit_hash in hashes_to_reword and action == 'pick':
                logging.info(f"Changing 'pick {{commit_hash}}' to 'reword {{commit_hash}}'")
                new_line = f'r {{commit_hash}}{{rest_of_line}}\\n' # Use 'r' for reword
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
             logging.warning(f"Could not parse todo line: {{stripped_line}}")
             new_lines.append(line)

    logging.info(f"Writing {{len(new_lines)}} lines back to {{todo_file_path}}")
    with open(todo_file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    logging.info("GIT_SEQUENCE_EDITOR script finished successfully.")
    sys.exit(0)

except Exception as e:
    logging.error(f"Error in GIT_SEQUENCE_EDITOR script: {{e}}", exc_info=True)
    sys.exit(1)
"""
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        logging.info(f"Created GIT_SEQUENCE_EDITOR script: {script_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to create GIT_SEQUENCE_EDITOR script: {e}")
        return False


def create_rebase_commit_editor_script(script_path):
    """Creates the python script for GIT_EDITOR (provides new commit message)."""
    # Note: reword_plan_json is a JSON string containing the {hash: new_message} mapping
    script_content = f"""#!/usr/bin/env python3
import sys
import logging
import re
import os
import subprocess
import json

logging.basicConfig(level=logging.WARN, format="%(levelname)s: %(message)s")

commit_msg_file_path = sys.argv[1]
logging.info(f"GIT_EDITOR script started for commit message file: {{commit_msg_file_path}}")

# The reword plan (hash -> new_message) is passed via environment variable as JSON
reword_plan_json = os.environ.get('GIT_REWORD_PLAN')
if not reword_plan_json:
    logging.error("GIT_REWORD_PLAN environment variable not set.")
    sys.exit(1)

try:
    reword_plan = json.loads(reword_plan_json)
    logging.info(f"Loaded reword plan for {{len(reword_plan)}} commits.")
except json.JSONDecodeError as e:
    logging.error(f"Failed to decode GIT_REWORD_PLAN JSON: {{e}}")
    sys.exit(1)

# --- How to identify the current commit being reworded? ---
# This is the tricky part. Git doesn't directly tell the editor which commit
# it's editing during a reword.
# Approach 1: Read the *original* message from the file Git provides.
#             Extract the original hash (if possible, maybe from a trailer?). Unreliable.
# Approach 2: Rely on the *order*. Requires knowing the rebase todo list order. Fragile.
# Approach 3: Use `git rev-parse HEAD`? Might work if HEAD points to the commit being edited. Needs testing.
# Approach 4: Pass the *current* target hash via another env var set by the main script
#             before calling rebase? Seems overly complex.

# --- Let's try Approach 3 (Check HEAD) ---
try:
    # Use subprocess to run git command to get the full hash of HEAD
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, encoding='utf-8')
    current_full_hash = result.stdout.strip()
    logging.info(f"Current HEAD full hash: {{current_full_hash}}")

    # Find the corresponding short hash in our plan (keys are short hashes)
    current_short_hash = None
    for short_h in reword_plan.keys():
        # Use git rev-parse to check if short_h resolves to current_full_hash
        # This handles potential ambiguity if multiple commits have the same short hash prefix
        try:
            # Verify that the short_h from the plan resolves to a commit object
            # and get its full hash. Simply pass the short hash to verify.
            logging.info(f"Verifying short hash {{short_h}} against HEAD {{current_full_hash}}...")
            verify_result = subprocess.run(['git', 'rev-parse', '--verify', short_h], capture_output=True, text=True, check=True, encoding='utf-8')
            verified_full_hash = verify_result.stdout.strip()
            if verified_full_hash == current_full_hash:
                current_short_hash = short_h
                logging.info(f"Matched HEAD {{current_full_hash}} to short hash {{current_short_hash}} in plan.")
                break
        except subprocess.CalledProcessError:
            logging.debug(f"Short hash {{short_h}} does not resolve to HEAD.")
            continue # Try next short hash in plan

    if current_short_hash is None:
        sys.exit(0) # Exit successfully to avoid blocking rebase, but log warning
    elif current_short_hash and current_short_hash in reword_plan:
        new_message = reword_plan[current_short_hash]
        logging.info(f"Found new message for commit {{current_short_hash}}.")
        # Remove the "Subject: " prefix as Git adds that structure
        new_message_content = re.sub(r"^[Ss]ubject:\s*", "", new_message, count=1)

        logging.info(f"Writing new message to {{commit_msg_file_path}}: {{new_message_content[:100]}}...")
        with open(commit_msg_file_path, 'w', encoding='utf-8') as f:
            f.write(new_message_content)

        logging.info("GIT_EDITOR script finished successfully for reword.")
        sys.exit(0)
    else:
        logging.warning(f"Could not find a matching commit hash in the reword plan for current HEAD {{current_full_hash}} (Short hash: {{current_short_hash}}).")
        # Keep the original message provided by Git? Or fail? Let's keep original for safety.
        logging.warning("Keeping the original commit message.")
        sys.exit(0) # Exit successfully to avoid blocking rebase, but log warning

except subprocess.CalledProcessError as e:
     logging.error(f"Failed to run git rev-parse HEAD: {{e}}")
     sys.exit(1) # Fail editor script
except Exception as e:
    logging.error(f"Error in GIT_EDITOR script: {{e}}", exc_info=True)
    sys.exit(1) # Exit with error code
"""
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        logging.info(f"Created GIT_EDITOR script: {script_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to create GIT_EDITOR script: {e}")
        return False


def attempt_auto_reword(merge_base, reword_plan):
    """Attempts to perform the rebase automatically applying rewording."""
    if not reword_plan:
        logging.info("No reword suggestions provided by AI. Skipping auto-rebase.")
        return True

    temp_dir = tempfile.mkdtemp(prefix="git_reword_")
    seq_editor_script_path = os.path.join(temp_dir, "rebase_sequence_editor.py")
    commit_editor_script_path = os.path.join(temp_dir, "rebase_commit_editor.py")
    logging.debug(f"Temporary directory: {temp_dir}")

    try:
        # Create the sequence editor script (changes pick -> reword)
        if not create_rebase_sequence_editor_script(
            seq_editor_script_path, reword_plan
        ):
            return False

        # Create the commit editor script (provides new message)
        # Pass the reword plan as a JSON string via environment variable
        reword_plan_json = json.dumps(reword_plan)
        if not create_rebase_commit_editor_script(commit_editor_script_path):
            return False

        # Prepare environment for the git command
        rebase_env = os.environ.copy()
        rebase_env["GIT_SEQUENCE_EDITOR"] = seq_editor_script_path
        rebase_env["GIT_EDITOR"] = commit_editor_script_path
        # Pass the plan to the commit editor script via env var
        rebase_env["GIT_REWORD_PLAN"] = reword_plan_json
        logging.debug(f"GIT_REWORD_PLAN: {reword_plan_json}")

        print("\nAttempting automatic rebase with suggested rewording...")
        logging.info(f"Running: git rebase -i {merge_base}")

        rebase_result = run_git_command(
            ["rebase", "-i", merge_base],
            check=False,
            capture_output=False,
            env=rebase_env,
        )

        if rebase_result is not None:
            print("✅ Automatic reword rebase completed successfully.")
            logging.info("Automatic reword rebase seems successful.")
            return True
        else:
            print("\n❌ Automatic reword rebase failed.")
            print(
                "   This could be due to merge conflicts, script errors, or other rebase issues."
            )
            logging.warning("Automatic reword rebase failed. Aborting...")

            print("   Attempting to abort the failed rebase (`git rebase --abort`)...")
            abort_result = run_git_command(
                ["rebase", "--abort"], check=False, capture_output=False
            )
            if abort_result is not None:
                print(
                    "   Rebase aborted successfully. Your branch is back to its original state."
                )
                logging.info("Failed rebase aborted successfully.")
            else:
                print("   ⚠️ Failed to automatically abort the rebase.")
                print("      Please run `git rebase --abort` manually to clean up.")
                logging.error("Failed to automatically abort the rebase.")
            return False

    except Exception as e:
        logging.error(
            f"An unexpected error occurred during auto-reword attempt: {e}",
            exc_info=True,
        )
        print("\n❌ An unexpected error occurred during the automatic reword attempt.")
        print(
            "   You may need to manually check your Git status and potentially run `git rebase --abort`."
        )
        return False
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil

                shutil.rmtree(temp_dir)
                logging.debug(f"Cleaned up temporary directory: {temp_dir}")
            except OSError as e:
                logging.warning(f"Could not remove temporary directory {temp_dir}: {e}")


# --- Main Execution ---


def main():
    """Main function to orchestrate Git analysis and AI interaction."""
    parser = argparse.ArgumentParser(
        description="Uses Gemini AI to suggest and automatically attempt Git 'reword' operations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "upstream_ref",
        nargs="?",
        default="upstream/main",
        help="The upstream reference point or commit hash to compare against.",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Only show AI suggestions and instructions; disable automatic reword attempt.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging."
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose logging enabled.")

    if not check_git_repository():
        logging.error("This script must be run from within a Git repository.")
        sys.exit(1)

    current_branch = get_current_branch()
    if not current_branch:
        logging.error("Could not determine the current Git branch.")
        sys.exit(1)
    logging.info(f"Current branch: {current_branch}")

    upstream_ref = args.upstream_ref
    logging.info(f"Comparing against reference: {upstream_ref}")

    # --- Safety: Create Backup Branch ---
    backup_branch = create_backup_branch(current_branch)
    if not backup_branch:
        try:
            confirm = input(
                "⚠️ Failed to create backup branch. Continue without backup? (yes/no): "
            ).lower()
        except EOFError:
            confirm = "no"
        if confirm != "yes":
            logging.info("Aborting.")
            sys.exit(1)
        else:
            logging.warning("Proceeding without a backup branch. Be careful!")
    else:
        print("-" * 40)
        print(f"✅ Backup branch created: {backup_branch}")
        print(
            f"   Restore with: git checkout {current_branch} && git reset --hard {backup_branch}"
        )
        print("-" * 40)

    # --- Gather Git Context ---
    print("\nGathering Git context...")
    commit_range, merge_base = get_commit_range(upstream_ref, current_branch)
    if not commit_range:
        sys.exit(1)

    logging.info(f"Analyzing commit range: {commit_range} (Merge Base: {merge_base})")

    commits_data = get_commits_in_range(commit_range)
    if not commits_data:
        logging.info(
            f"No commits found between '{merge_base}' and '{current_branch}'. Nothing to do."
        )
        sys.exit(0)

    diff = get_diff_in_range(commit_range)  # Diff might help AI judge messages

    # --- Interact with AI ---
    print("\nGenerating prompt for AI reword suggestions...")
    initial_prompt = generate_reword_suggestion_prompt(
        commit_range, merge_base, commits_data, diff
    )

    logging.debug("\n--- Initial AI Prompt Snippet ---")
    logging.debug(initial_prompt[:1000] + "...")
    logging.debug("--- End Prompt Snippet ---\n")

    print(f"Sending request to Gemini AI ({MODEL_NAME})...")

    ai_response_text = ""
    reword_suggestions_text = ""  # Store raw AI suggestions
    try:
        # For reword, file content is less likely needed, but keep structure just in case
        convo = model.start_chat(history=[])
        response = convo.send_message(initial_prompt)
        ai_response_text = response.text

        # Store the final AI response containing suggestions
        reword_suggestions_text = ai_response_text.strip()

        # Parse the suggestions
        reword_plan = parse_reword_suggestions(reword_suggestions_text, commits_data)

        if not reword_plan:
            print("\n💡 AI did not suggest any specific reword operations.")
        else:
            print("\n💡 --- AI Reword Suggestions --- 💡")
            for i, (hash_key, msg) in enumerate(reword_plan.items()):
                print(f"  {i + 1}. Reword commit `{hash_key}` with new message:")
                # Indent the message for readability
                indented_msg = "     " + msg.replace("\n", "\n     ")
                print(indented_msg)
                print("-" * 20)  # Separator
            print("💡 --- End AI Suggestions --- 💡")

        # --- Attempt Automatic Rebase or Show Instructions ---
        if not args.instruct:  # Default behavior: attempt auto-reword
            if reword_plan:
                success = attempt_auto_reword(merge_base, reword_plan)
                if not success:
                    # Failure message already printed by attempt_auto_reword
                    print("\n" + "=" * 60)
                    print("🛠️ MANUAL REBASE REQUIRED 🛠️")
                    print("=" * 60)
                    print("The automatic reword rebase failed.")
                    print("Please perform the rebase manually:")
                    print(f"  1. Run: `git rebase -i {merge_base}`")
                    print(
                        "  2. In the editor, change 'pick' to 'r' (or 'reword') for the commits"
                    )
                    print("     suggested by the AI above.")
                    print(
                        "  3. Save the editor. Git will stop at each commit marked for reword."
                    )
                    print(
                        "  4. Manually replace the old commit message with the AI-suggested one:"
                    )
                    print("     ```text")
                    # Print raw suggestions which might be easier to copy/paste
                    print(
                        reword_suggestions_text
                        if reword_suggestions_text
                        else "     (No specific reword suggestions found in AI response)"
                    )
                    print("     ```")
                    print(
                        "  5. Save the message editor and continue the rebase (`git rebase --continue`)."
                    )
                    if backup_branch:
                        print(f"  6. Remember backup branch: {backup_branch}")
                    print("=" * 60)
                    sys.exit(1)  # Exit with error status after failure
                else:
                    # Auto reword succeeded
                    print("\nBranch history has been modified by automatic rewording.")
                    if backup_branch:
                        print(
                            f"Backup branch '{backup_branch}' still exists if needed."
                        )
            else:
                print("\nNo automatic rebase attempted as AI suggested no rewording.")

        elif reword_plan:  # --instruct flag was used AND suggestions exist
            print("\n" + "=" * 60)
            print("📝 MANUAL REBASE INSTRUCTIONS (--instruct used) 📝")
            print("=" * 60)
            print("AI suggested the rewording listed above.")
            print("To apply them manually:")
            print(f"  1. Run: `git rebase -i {merge_base}`")
            print(
                "  2. Edit the 'pick' lines in the editor, changing 'pick' to 'r' (or 'reword')"
            )
            print("     for the commits listed above.")
            print(
                "  3. Save the editor. Git will stop at each commit marked for reword."
            )
            print(
                "  4. Manually replace the old commit message with the corresponding AI-suggested message."
            )
            print(
                "  5. Save the message editor and continue the rebase (`git rebase --continue`)."
            )
            if backup_branch:
                print(f"  6. Remember backup branch: {backup_branch}")
            print("=" * 60)

    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {e}", exc_info=True)
        try:  # Log AI feedback if possible
            if response and hasattr(response, "prompt_feedback"):
                logging.error(f"AI Prompt Feedback: {response.prompt_feedback}")
            # ... (rest of feedback logging) ...
        except Exception as feedback_e:
            logging.error(
                f"Could not retrieve detailed feedback from AI response: {feedback_e}"
            )
        print("\n❌ An unexpected error occurred during the process.")
        print("   Please check the logs and your Git status.")
        print("   You may need to run `git rebase --abort` manually.")


if __name__ == "__main__":
    main()
