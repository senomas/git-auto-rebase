import subprocess
import google.generativeai as genai
import os
import argparse
import sys
import datetime
import re
import logging
import tempfile

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
    logging.error("and set it as an environment variable:")
    logging.error("  export GEMINI_API_KEY='YOUR_API_KEY'  (Linux/macOS)")
    logging.error("  set GEMINI_API_KEY=YOUR_API_KEY       (Windows CMD)")
    logging.error("  $env:GEMINI_API_KEY='YOUR_API_KEY'   (Windows PowerShell)")
    sys.exit(1)

# Configure the Gemini AI Client
try:
    genai.configure(api_key=API_KEY)
    # Use a model suitable for complex reasoning like code analysis.
    # Adjust model name if needed (e.g., 'gemini-1.5-flash-latest').
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

# --- Git Helper Functions ---


def run_git_command(command_list, check=True, capture_output=True, env=None):
    """
    Runs a Git command as a list of arguments and returns its stdout.
    Handles errors and returns None on failure if check=True.
    Allows passing environment variables.
    """
    full_command = []
    try:
        full_command = ["git"] + command_list
        logging.debug(f"Running command: {' '.join(full_command)}")
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
        logging.debug(f"Command successful. Output:\n{result.stdout[:200]}...")
        return result.stdout.strip() if capture_output else ""
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
        logging.error(f"An unexpected error occurred running git: {e}")
        return None


def check_git_repository():
    """Checks if the current directory is the root of a Git repository."""
    output = run_git_command(["rev-parse", "--is-inside-work-tree"])
    return output == "true"


def get_current_branch():
    """Gets the current active Git branch name."""
    return run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])


def create_backup_branch(branch_name):
    """Creates a timestamped backup branch from the given branch name."""
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
        logging.error("Failed to create backup branch.")
        return None


def get_commit_range(upstream_ref, current_branch):
    """
    Determines the commit range (merge_base..current_branch).
    Returns the range string and the merge base hash.
    """
    logging.info(
        f"Finding merge base between '{upstream_ref}' and '{current_branch}'..."
    )
    merge_base = run_git_command(["merge-base", upstream_ref, current_branch])
    if not merge_base:
        logging.error(
            f"Could not find merge base between '{upstream_ref}' and '{current_branch}'."
        )
        logging.error(
            f"Ensure '{upstream_ref}' is a valid reference (branch, commit, tag)"
        )
        logging.error("and that it has been fetched (e.g., 'git fetch origin').")
        return None, None  # Indicate failure

    logging.info(f"Found merge base: {merge_base}")
    commit_range = f"{merge_base}..{current_branch}"
    return commit_range, merge_base


def get_commits_in_range(commit_range):
    """Gets a list of commit hashes and subjects in the specified range (oldest first)."""
    log_output = run_git_command(
        ["log", "--pretty=format:%h %s", "--reverse", commit_range]
    )
    if log_output is not None:
        commits = log_output.splitlines()
        logging.info(f"Found {len(commits)} commits in range {commit_range}.")
        return commits
    return []  # Return empty list on failure or no commits


def get_changed_files_in_range(commit_range):
    """
    Gets a list of files changed in the specified range and generates
    a simple directory structure string representation.
    """
    diff_output = run_git_command(["diff", "--name-only", commit_range])
    if diff_output is not None:
        files = diff_output.splitlines()
        logging.info(f"Found {len(files)} changed files in range {commit_range}.")

        # Basic tree structure representation
        tree = {}
        for file_path in files:
            parts = file_path.replace("\\", "/").split("/")
            node = tree
            for i, part in enumerate(parts):
                if not part:
                    continue
                if i == len(parts) - 1:
                    node[part] = "file"
                else:
                    if part not in node:
                        node[part] = {}
                    if isinstance(node[part], dict):
                        node = node[part]
                    else:
                        logging.warning(
                            f"Path conflict building file tree for: {file_path}"
                        )
                        break

        def format_tree(d, indent=0):
            lines = []
            for key, value in sorted(d.items()):
                prefix = "  " * indent
                if isinstance(value, dict):
                    lines.append(f"{prefix}📁 {key}/")
                    lines.extend(format_tree(value, indent + 1))
                else:
                    lines.append(f"{prefix}📄 {key}")
            return lines

        tree_str = "\n".join(format_tree(tree))
        return tree_str, files
    return "", []


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


def get_file_content_at_commit(commit_hash, file_path):
    """Gets the content of a specific file at a specific commit hash."""
    logging.info(f"Fetching content of '{file_path}' at commit {commit_hash[:7]}...")
    content = run_git_command(["show", f"{commit_hash}:{file_path}"])
    if content is None:
        logging.warning(
            f"Could not retrieve content for {file_path} at {commit_hash[:7]}."
        )
        return None
    return content


# --- AI Interaction ---


def generate_fixup_suggestion_prompt(
    commit_range, merge_base, commits, file_structure, diff
):
    """
    Creates a prompt asking the AI specifically to identify potential
    fixup candidates within the commit range.
    Returns suggestions in a parsable format.
    """

    commit_list_str = (
        "\n".join([f"- {c}" for c in commits]) if commits else "No commits in range."
    )

    prompt = f"""
You are an expert Git assistant. Your task is to analyze the provided Git commit history and identify commits within the range `{commit_range}` that should be combined using `fixup` during an interactive rebase (`git rebase -i {merge_base}`).

**Goal:** Identify commits that are minor corrections or direct continuations of the immediately preceding commit, where the commit message can be discarded.

**Git Commit Message Conventions (for context):**
* Subject: Imperative, < 50 chars, capitalized, no period. Use types like `feat:`, `fix:`, `refactor:`, etc.
* Body: Explain 'what' and 'why', wrap at 72 chars.

**Provided Context:**

1.  **Commit Range:** `{commit_range}`
2.  **Merge Base Hash:** `{merge_base}`
3.  **Commits in Range (Oldest First - Short Hash & Subject):**
```
{commit_list_str}
```
4.  **Changed Files Structure in Range:**
```
{file_structure if file_structure else "No files changed or unable to list."}
```
5.  **Combined Diff for the Range (`git diff --patch-with-stat {commit_range}`):**
```diff
{diff if diff else "No differences found or unable to get diff."}
```

**Instructions:**

1.  Analyze the commits, their messages, the changed files, and the diff.
2.  Identify commits from the list that are strong candidates for being combined into their **immediately preceding commit** using `fixup` (combine changes, discard message). Focus on small fixes, typo corrections, or direct continuations where the commit message isn't valuable.
3.  For each suggestion, output *only* a line in the following format:
    `FIXUP: <hash_to_fixup> INTO <preceding_hash>`
    Use the short commit hashes provided in the commit list.
4.  Provide *only* lines in the `FIXUP:` format. Do not include explanations, introductory text, or any other formatting. If no fixups are suggested, output nothing.

**Example Output:**

```text
FIXUP: hash2 INTO hash1
FIXUP: hash5 INTO hash4
```

5.  **File Content Request:** If you absolutely need the content of specific files *at specific commits* to confidently determine if they should be fixed up, ask for them clearly ONCE. List the files using this exact format at the end of your response:
    `REQUEST_FILES: [commit_hash1:path/to/file1.py, commit_hash2:another/path/file2.js]`
    Use the short commit hashes provided in the commit list. Do *not* ask for files unless essential for *this specific task* of identifying fixup candidates.

Now, analyze the provided context and generate *only* the `FIXUP:` lines or `REQUEST_FILES:` line.
"""
    return prompt


def parse_fixup_suggestions(ai_response_text, commits_in_range):
    """Parses AI response for FIXUP: lines and validates hashes."""
    fixup_pairs = []
    commit_hashes = {
        c.split()[0] for c in commits_in_range
    }  # Set of valid short hashes

    for line in ai_response_text.splitlines():
        line = line.strip()
        if line.startswith("FIXUP:"):
            match = re.match(r"FIXUP:\s*(\w+)\s+INTO\s+(\w+)", line, re.IGNORECASE)
            if match:
                fixup_hash = match.group(1)
                target_hash = match.group(2)
                # Validate that both hashes were in the original commit list
                if fixup_hash in commit_hashes and target_hash in commit_hashes:
                    fixup_pairs.append({"fixup": fixup_hash, "target": target_hash})
                    logging.debug(
                        f"Parsed fixup suggestion: {fixup_hash} into {target_hash}"
                    )
                else:
                    logging.warning(
                        f"Ignoring invalid fixup suggestion (hash not in range): {line}"
                    )
            else:
                logging.warning(f"Could not parse FIXUP line: {line}")
    return fixup_pairs


# --- request_files_from_user function remains the same ---
def request_files_from_user(requested_files_str, commits_in_range):
    """
    Parses AI request string "REQUEST_FILES: [hash:path, ...]", verifies hashes,
    asks user permission, fetches file contents, and returns formatted context.
    """
    file_requests = []
    try:
        content_match = re.search(
            r"REQUEST_FILES:\s*\[(.*)\]", requested_files_str, re.IGNORECASE | re.DOTALL
        )
        if not content_match:
            logging.warning("Could not parse file request format from AI response.")
            return None, None

        items_str = content_match.group(1).strip()
        if not items_str:
            logging.info("AI requested files but the list was empty.")
            return None, None

        items = [item.strip() for item in items_str.split(",") if item.strip()]
        commit_hash_map = {c.split()[0]: c.split()[0] for c in commits_in_range}

        for item in items:
            if ":" not in item:
                logging.warning(
                    f"Invalid format in requested file item (missing ':'): {item}"
                )
                continue
            commit_hash, file_path = item.split(":", 1)
            commit_hash = commit_hash.strip()
            file_path = file_path.strip()

            if commit_hash not in commit_hash_map:
                logging.warning(
                    f"AI requested file for unknown/out-of-range commit hash '{commit_hash}'. Skipping."
                )
                continue
            file_requests.append({"hash": commit_hash, "path": file_path})

    except Exception as e:
        logging.error(f"Error parsing requested files string: {e}")
        return None, None

    if not file_requests:
        logging.info("No valid file requests found after parsing AI response.")
        return None, None

    print("\n----------------------------------------")
    print("❓ AI Request for File Content ❓")
    print("----------------------------------------")
    print("The AI needs the content of the following files at specific commits")
    print("to provide more accurate fixup suggestions:")
    files_to_fetch = []
    for i, req in enumerate(file_requests):
        print(f"  {i + 1}. File: '{req['path']}' at commit {req['hash']}")
        files_to_fetch.append(req)

    if not files_to_fetch:
        print("\nNo valid files to fetch based on the request.")
        return None, None

    print("----------------------------------------")

    while True:
        try:
            answer = (
                input("Allow fetching these file contents? (yes/no): ").lower().strip()
            )
        except EOFError:
            logging.warning("Input stream closed. Assuming 'no'.")
            answer = "no"

        if answer == "yes":
            logging.info("User approved fetching file content.")
            fetched_content_list = []
            for req in files_to_fetch:
                content = get_file_content_at_commit(req["hash"], req["path"])
                if content is not None:
                    fetched_content_list.append(
                        f"--- Content of '{req['path']}' at commit {req['hash']} ---\n"
                        f"```\n{content}\n```\n"
                        f"--- End Content for {req['path']} at {req['hash']} ---"
                    )
                else:
                    fetched_content_list.append(
                        f"--- Could not fetch content of '{req['path']}' at commit {req['hash']} ---"
                    )
            return "\n\n".join(fetched_content_list), requested_files_str

        elif answer == "no":
            logging.info("User denied fetching file content.")
            return None, requested_files_str
        else:
            print("Please answer 'yes' or 'no'.")


# --- Automatic Rebase Logic ---


def create_rebase_editor_script(script_path, fixup_plan):
    """Creates the python script to be used by GIT_SEQUENCE_EDITOR."""
    # Create a set of hashes that need to be fixed up
    fixups_to_apply = {pair["fixup"] for pair in fixup_plan}

    script_content = f"""#!/usr/bin/env python3
import sys
import logging
import re
import os

# Define log file path relative to the script itself
log_file = __file__ + ".log"
# Setup logging within the editor script to write to the log file
logging.basicConfig(filename=log_file, filemode='w', level=logging.WARN, format="%(asctime)s - %(levelname)s: %(message)s")

todo_file_path = sys.argv[1]
logging.info(f"GIT_SEQUENCE_EDITOR script started for: {{todo_file_path}}")

# Hashes that should be changed to 'fixup'
fixups_to_apply = {fixups_to_apply!r}
logging.info(f"Applying fixups for hashes: {{fixups_to_apply}}")

new_lines = []
try:
    with open(todo_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        stripped_line = line.strip()
        # Skip comments and blank lines
        if not stripped_line or stripped_line.startswith('#'):
            new_lines.append(line)
            continue

        # Use regex for more robust parsing of todo lines (action hash ...)
        match = re.match(r"^(\w+)\s+([0-9a-fA-F]+)(.*)", stripped_line)
        if match:
            action = match.group(1).lower()
            commit_hash = match.group(2)
            rest_of_line = match.group(3)

            # Check if this commit should be fixed up
            if commit_hash in fixups_to_apply and action == 'pick':
                logging.info(f"Changing 'pick {{commit_hash}}' to 'fixup {{commit_hash}}'")
                # Replace 'pick' with 'fixup', preserving the rest of the line
                new_line = f'f {{commit_hash}}{{rest_of_line}}\\n'
                new_lines.append(new_line)
            else:
                # Keep the original line
                new_lines.append(line)
        else:
             # Keep lines that don't look like standard todo lines
             logging.warning(f"Could not parse todo line: {{stripped_line}}")
             new_lines.append(line)


    logging.info(f"Writing {{len(new_lines)}} lines back to {{todo_file_path}}")
    with open(todo_file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    logging.info("GIT_SEQUENCE_EDITOR script finished successfully.")
    sys.exit(0) # Explicitly exit successfully

except Exception as e:
    logging.error(f"Error in GIT_SEQUENCE_EDITOR script: {{e}}", exc_info=True)
    sys.exit(1) # Exit with error code
"""
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        # Make the script executable (important on Linux/macOS)
        os.chmod(script_path, 0o755)
        logging.info(f"Created GIT_SEQUENCE_EDITOR script: {script_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to create GIT_SEQUENCE_EDITOR script: {e}")
        return False


def attempt_auto_fixup(merge_base, fixup_plan):
    """Attempts to perform the rebase automatically applying fixups."""
    if not fixup_plan:
        logging.info("No fixup suggestions provided by AI. Skipping auto-rebase.")
        return True  # Nothing to do, considered success

    # Use a temporary directory to hold the script and its log
    temp_dir = tempfile.mkdtemp(prefix="git_rebase_")
    editor_script_path = os.path.join(temp_dir, "rebase_editor.py")
    logging.debug(f"Temporary directory: {temp_dir}")
    logging.debug(f"Temporary editor script path: {editor_script_path}")

    try:
        if not create_rebase_editor_script(editor_script_path, fixup_plan):
            return False  # Failed to create script

        # Prepare environment for the git command
        rebase_env = os.environ.copy()
        rebase_env["GIT_SEQUENCE_EDITOR"] = editor_script_path
        # Prevent Git from opening a standard editor for messages etc.
        # 'true' simply exits successfully, accepting default messages
        rebase_env["GIT_EDITOR"] = "true"

        print("\nAttempting automatic rebase with suggested fixups...")
        logging.info(f"Running: git rebase -i {merge_base}")
        # Run rebase non-interactively, check=False to handle failures manually
        rebase_result = run_git_command(
            ["rebase", "-i", merge_base],
            check=False,  # Don't raise exception on failure, check exit code
            capture_output=True,  # Capture output to see potential errors
            env=rebase_env,
        )

        # Check the result (run_git_command returns None on CalledProcessError)
        if rebase_result is not None:
            # Command finished, exit code was likely 0 (success)
            print("✅ Automatic fixup rebase completed successfully.")
            logging.info("Automatic fixup rebase seems successful.")
            return True
        else:
            # Command failed (non-zero exit code, run_git_command returned None)
            print("\n❌ Automatic fixup rebase failed.")
            print(
                "   This likely means merge conflicts occurred or another rebase error happened."
            )
            logging.warning("Automatic fixup rebase failed. Aborting...")

            # Attempt to abort the failed rebase
            print("   Attempting to abort the failed rebase (`git rebase --abort`)...")
            # Run abort without capturing output, just check success/failure
            abort_result = run_git_command(
                ["rebase", "--abort"], check=False, capture_output=False
            )
            # run_git_command returns None on failure (CalledProcessError)
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
            f"An unexpected error occurred during auto-fixup attempt: {e}",
            exc_info=True,
        )
        # Might need manual cleanup here too
        print("\n❌ An unexpected error occurred during the automatic fixup attempt.")
        print(
            "   You may need to manually check your Git status and potentially run `git rebase --abort`."
        )
        return False
    finally:
        # Determine if rebase failed *before* potential cleanup errors
        # Note: rebase_result is defined in the outer scope of the try block
        rebase_failed = "rebase_result" in locals() and rebase_result is None

        # Check if we need to display the editor script log
        editor_log_path = editor_script_path + ".log"
        verbose_logging = logging.getLogger().isEnabledFor(logging.DEBUG)

        if (rebase_failed or verbose_logging) and os.path.exists(editor_log_path):
            try:
                with open(editor_log_path, "r", encoding="utf-8") as log_f:
                    log_content = log_f.read()
                if log_content:
                    print("\n--- Rebase Editor Script Log ---")
                    print(log_content.strip())
                    print("--- End Log ---")
                else:
                    # Only log if verbose, otherwise it's just noise
                    if verbose_logging:
                        logging.debug(
                            f"Rebase editor script log file was empty: {editor_log_path}"
                        )
            except Exception as log_e:
                logging.warning(
                    f"Could not read rebase editor script log file {editor_log_path}: {log_e}"
                )

        # Clean up the temporary directory and its contents
        if temp_dir and os.path.exists(temp_dir):
            try:
                if os.path.exists(editor_log_path):
                    os.remove(editor_log_path)
                if os.path.exists(editor_script_path):
                    os.remove(editor_script_path)
                os.rmdir(temp_dir)
                logging.debug(f"Cleaned up temporary directory: {temp_dir}")
            except OSError as e:
                logging.warning(
                    f"Could not completely remove temporary directory {temp_dir}: {e}"
                )


# --- Main Execution ---


def main():
    """Main function to orchestrate Git analysis and AI interaction."""
    parser = argparse.ArgumentParser(
        description="Uses Gemini AI to suggest and automatically attempt Git 'fixup' operations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "upstream_ref",
        nargs="?",
        default="upstream/main",
        help="The upstream reference point or commit hash to compare against "
        "(e.g., 'origin/main', 'upstream/develop', specific_commit_hash). "
        "Ensure this reference exists and is fetched.",
    )
    # --- Argument Change ---
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Only show AI suggestions and instructions; disable automatic fixup attempt.",
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
    # Always create backup, especially if attempting auto-rebase
    backup_branch = create_backup_branch(current_branch)
    if not backup_branch:
        try:
            confirm = input(
                "⚠️ Failed to create backup branch. Continue without backup? (yes/no): "
            ).lower()
        except EOFError:
            logging.warning("Input stream closed. Aborting.")
            confirm = "no"
        if confirm != "yes":
            logging.info("Aborting.")
            sys.exit(1)
        else:
            logging.warning("Proceeding without a backup branch. Be careful!")
    else:
        print("-" * 40)
        print(f"✅ Backup branch created: {backup_branch}")
        print("   If anything goes wrong, you can restore using:")
        print(f"     git checkout {current_branch}")
        print(f"     git reset --hard {backup_branch}")
        print("-" * 40)

    # --- Gather Git Context ---
    print("\nGathering Git context...")
    commit_range, merge_base = get_commit_range(upstream_ref, current_branch)
    if not commit_range:
        sys.exit(1)

    logging.info(f"Analyzing commit range: {commit_range} (Merge Base: {merge_base})")

    commits = get_commits_in_range(commit_range)
    if not commits:
        logging.info(
            f"No commits found between '{merge_base}' and '{current_branch}'. Nothing to do."
        )
        sys.exit(0)

    file_structure, changed_files_list = get_changed_files_in_range(commit_range)
    diff = get_diff_in_range(commit_range)

    if not diff and not changed_files_list:
        logging.warning(
            f"No file changes or diff found between '{merge_base}' and '{current_branch}',"
        )
        logging.warning("even though commits exist. AI suggestions might be limited.")
        # Don't exit automatically, let AI try

    # --- Interact with AI ---
    print("\nGenerating prompt for AI fixup suggestions...")
    initial_prompt = generate_fixup_suggestion_prompt(
        commit_range, merge_base, commits, file_structure, diff
    )

    logging.debug("\n--- Initial AI Prompt Snippet ---")
    logging.debug(initial_prompt[:1000] + "...")
    logging.debug("--- End Prompt Snippet ---\n")

    print(f"Sending request to Gemini AI ({MODEL_NAME})...")

    ai_response_text = ""
    fixup_suggestions_text = ""  # Store the raw suggestions for later display if needed
    try:
        convo = model.start_chat(history=[])
        response = convo.send_message(initial_prompt)
        ai_response_text = response.text

        # Loop for file requests
        while "REQUEST_FILES:" in ai_response_text.upper():
            logging.info("AI requested additional file content.")
            additional_context, original_request = request_files_from_user(
                ai_response_text, commits
            )

            if additional_context:
                logging.info("Sending fetched file content back to AI...")
                follow_up_prompt = f"""
Okay, here is the content of the files you requested:

{additional_context}

Please use this new information to refine your **fixup suggestions** based on the original request and context. Provide the final list of `FIXUP: ...` lines now. Remember to *only* suggest fixup actions and output *only* `FIXUP:` lines. Do not ask for more files.
"""
                logging.debug("\n--- Follow-up AI Prompt Snippet ---")
                logging.debug(follow_up_prompt[:500] + "...")
                logging.debug("--- End Follow-up Snippet ---\n")
                response = convo.send_message(follow_up_prompt)
                ai_response_text = response.text
            else:
                logging.info(
                    "Proceeding without providing files as requested by AI or user."
                )
                no_files_prompt = f"""
I cannot provide the content for the files you requested ({original_request}).
Please proceed with generating the **fixup suggestions** based *only* on the initial context (commit list, file structure, diff) I provided earlier. Make your best suggestions without the file content. Provide the final list of `FIXUP: ...` lines now. Remember to *only* suggest fixup actions.
"""
                logging.debug("\n--- No-Files AI Prompt ---")
                logging.debug(no_files_prompt)
                logging.debug("--- End No-Files Prompt ---\n")
                response = convo.send_message(no_files_prompt)
                ai_response_text = response.text
                break

        # Store the final AI response containing suggestions
        fixup_suggestions_text = ai_response_text.strip()

        # Parse the suggestions
        fixup_plan = parse_fixup_suggestions(fixup_suggestions_text, commits)

        if not fixup_plan:
            print("\n💡 AI did not suggest any specific fixup operations.")
        else:
            print("\n💡 --- AI Fixup Suggestions --- 💡")
            # Print the parsed plan for clarity
            for i, pair in enumerate(fixup_plan):
                print(
                    f"  {i + 1}. Fixup commit `{pair['fixup']}` into `{pair['target']}`"
                )
            print("💡 --- End AI Suggestions --- 💡")

        # --- Attempt Automatic Rebase or Show Instructions ---
        # --- Logic Change ---
        if not args.instruct:  # Default behavior: attempt auto-fixup
            if fixup_plan:
                success = attempt_auto_fixup(merge_base, fixup_plan)
                if not success:
                    # Failure message already printed by attempt_auto_fixup
                    print("\n" + "=" * 60)
                    print("🛠️ MANUAL REBASE REQUIRED 🛠️")
                    print("=" * 60)
                    print(
                        "The automatic fixup rebase failed (likely due to conflicts)."
                    )
                    print("Please perform the rebase manually:")
                    print(f"  1. Run: `git rebase -i {merge_base}`")
                    print(
                        "  2. In the editor, change 'pick' to 'f' (or 'fixup') for the commits"
                    )
                    print(
                        "     suggested by the AI above (and any other changes you want)."
                    )
                    print("     Original AI suggestions:")
                    print("     ```text")
                    # Print raw suggestions which might be easier to copy/paste
                    print(
                        fixup_suggestions_text
                        if fixup_suggestions_text
                        else "     (No specific fixup lines found in AI response)"
                    )
                    print("     ```")
                    print("  3. Save the editor and resolve any conflicts Git reports.")
                    print(
                        "     Use `git status`, edit files, `git add <files>`, `git rebase --continue`."
                    )
                    if backup_branch:
                        print(f"  4. Remember backup branch: {backup_branch}")
                    print("=" * 60)
                    sys.exit(1)  # Exit with error status after failure
                else:
                    # Auto fixup succeeded
                    print("\nBranch history has been modified by automatic fixups.")
                    if backup_branch:
                        print(
                            f"Backup branch '{backup_branch}' still exists if needed."
                        )
            else:
                print("\nNo automatic rebase attempted as AI suggested no fixups.")

        elif fixup_plan:  # --instruct flag was used AND suggestions exist
            print("\n" + "=" * 60)
            print("📝 MANUAL REBASE INSTRUCTIONS (--instruct used) 📝")
            print("=" * 60)
            print("AI suggested the fixups listed above.")
            print("To apply them (or other changes):")
            print(f"  1. Run: `git rebase -i {merge_base}`")
            print("  2. Edit the 'pick' lines in the editor based on the suggestions")
            print("     (changing 'pick' to 'f' or 'fixup').")
            print("  3. Save the editor and follow Git's instructions.")
            if backup_branch:
                print(f"  4. Remember backup branch: {backup_branch}")
            print("=" * 60)
        # If --instruct and no fixup_plan, nothing specific needs to be printed here

    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {e}", exc_info=True)
        # Attempt to print feedback if available
        try:
            if response and hasattr(response, "prompt_feedback"):
                logging.error(f"AI Prompt Feedback: {response.prompt_feedback}")
            if response and hasattr(response, "candidates"):
                for candidate in response.candidates:
                    logging.error(
                        f"AI Candidate Finish Reason: {candidate.finish_reason}"
                    )
                    if hasattr(candidate, "safety_ratings"):
                        logging.error(f"AI Safety Ratings: {candidate.safety_ratings}")
        except Exception as feedback_e:
            logging.error(
                f"Could not retrieve detailed feedback from AI response: {feedback_e}"
            )
        print("\n❌ An unexpected error occurred during the process.")
        print("   Please check the logs and your Git status.")
        print("   You may need to run `git rebase --abort` manually.")


if __name__ == "__main__":
    main()
