"""Push instructions to the results branch for the Colab notebook to read.

Usage:
  python instruct.py <action> [key=value ...]
  
Examples:
  python instruct.py retry n_trees=3000 k_best=800
  python instruct.py notify message="try different learning rate"
  python instruct.py stop
  python instruct.py show
"""
import sys, json, base64, subprocess, time, tempfile, os

OWNER = "Hanishchow"
REPO = "gecokkkkkkkkq"
BRANCH = "results"
PATH = "instructions.json"

def gh_api(method, endpoint, body=None):
    if body:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(body, f)
            tmpfile = f.name
        try:
            cmd = ["gh", "api", "--method", method, endpoint, "--input", tmpfile]
            result = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            os.unlink(tmpfile)
    else:
        cmd = ["gh", "api", "--method", method, endpoint]
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        err = result.stderr.strip()
        if "Not Found" in err and method == "GET":
            return None
        print(f"gh api error: {err}")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return result.stdout

def get_file_sha():
    data = gh_api("GET", f"/repos/{OWNER}/{REPO}/contents/{PATH}?ref={BRANCH}")
    if data and isinstance(data, dict) and "sha" in data:
        return data["sha"]
    return None

def push_instruction(action, params):
    content = {
        "action": action,
        "params": params,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    encoded = base64.b64encode(json.dumps(content).encode()).decode()
    
    body = {
        "message": f"instruction: {action}",
        "content": encoded,
        "branch": BRANCH
    }
    sha = get_file_sha()
    if sha:
        body["sha"] = sha
    
    result = gh_api("PUT", f"/repos/{OWNER}/{REPO}/contents/{PATH}", body)
    if result and "content" in result:
        print(f"OK: action='{action}' pushed to {BRANCH}/{PATH}")
    else:
        print(f"Failed: {result}")

def show_instruction():
    data = gh_api("GET", f"/repos/{OWNER}/{REPO}/contents/{PATH}?ref={BRANCH}")
    if data and isinstance(data, dict) and "content" in data:
        decoded = base64.b64decode(data["content"]).decode()
        print(json.dumps(json.loads(decoded), indent=2))
    else:
        print("No instruction found on results branch")

def parse_params(args):
    """Parse key=value arguments into dict."""
    params = {}
    for arg in args:
        if '=' in arg:
            key, val = arg.split('=', 1)
            # Try to parse as number
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass  # keep as string
            params[key] = val
    return params

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    if sys.argv[1] == "show":
        show_instruction()
        return
    
    action = sys.argv[1]
    params = parse_params(sys.argv[2:])
    
    push_instruction(action, params)

if __name__ == "__main__":
    main()
