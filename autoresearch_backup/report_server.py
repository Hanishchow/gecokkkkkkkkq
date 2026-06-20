"""Report server with Serveo SSH tunnel (no account needed).
Usage:
  python report_server.py [port]

The server starts on localhost:port, then creates a Serveo tunnel 
so Colab can reach it at a public URL.
"""
import sys, json, time, subprocess, threading, os, signal, socket
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()
reports = []
instructions = []
tunnel_proc = None

class ReportData(BaseModel):
    data: dict

class Instruction(BaseModel):
    action: str
    params: dict = {}

@app.post("/report/{cell_name}")
def post_report(cell_name: str, r: ReportData):
    entry = {"cell": cell_name, "time": time.strftime("%H:%M:%S"), **r.data}
    reports.append(entry)
    print(f"\n{'='*60}")
    print(f"[REPORT] {cell_name} @ {entry['time']}")
    for k, v in r.data.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")
    return {"ok": True, "total_reports": len(reports)}

@app.get("/instructions")
def get_instructions():
    if instructions:
        inst = instructions.pop(0)
        print(f"\n[INSTRUCTION SENT] {inst['action']} {inst.get('params', {})}")
        return {"instruction": inst}
    return {"instruction": None}

@app.post("/instructions")
def add_instruction(inst: Instruction):
    instructions.append({"action": inst.action, "params": inst.params})
    print(f"\n[INSTRUCTION QUEUED] {inst.action} {inst.params} (pending: {len(instructions)})")
    return {"ok": True, "pending": len(instructions)}

@app.get("/reports")
def list_reports():
    return {"count": len(reports), "reports": reports[-20:]}

@app.get("/health")
def health():
    return {"status": "ok", "reports": len(reports), "instructions_pending": len(instructions)}

def start_serveo_tunnel(port=8000):
    """Start Serveo SSH tunnel and return public URL."""
    global tunnel_proc
    try:
        cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-R", f"80:localhost:{port}",
            "serveo.net"
        ]
        print(f"\nStarting Serveo tunnel: {' '.join(cmd)}")
        tunnel_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        # Wait for tunnel URL to appear
        url = None
        start = time.time()
        while time.time() - start < 15:
            line = tunnel_proc.stdout.readline()
            if line:
                print(f"  tunnel: {line.strip()}")
                if "Forwarding HTTP traffic from" in line or "http://" in line.lower():
                    for word in line.split():
                        if word.startswith("http://") or word.startswith("https://"):
                            url = word.strip()
                            break
                if url:
                    break
        if url:
            print(f"\n{'='*60}")
            print(f"  SERVE.IO TUNNEL ACTIVE")
            print(f"  Public URL: {url}")
            print(f"{'='*60}")
            print(f"  Paste URL in Colab notebook cell:")
            print(f"  SERVER_URL = '{url}'")
            print(f"{'='*60}\n")
            return url
        else:
            print("[!] Could not detect Serveo tunnel URL from output")
            print("[!] Check the terminal output above for the URL")
            return None
    except FileNotFoundError:
        print("[!] SSH not found. Required for Serveo tunnel.")
        return None
    except Exception as e:
        print(f"[!] Serveo tunnel error: {e}")
        return None

def cleanup():
    if tunnel_proc:
        tunnel_proc.terminate()
        tunnel_proc.wait(timeout=5)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    print(f"\nStarting report server on 0.0.0.0:{port}...")
    
    # Start tunnel in separate thread
    threading.Thread(target=start_serveo_tunnel, args=(port,), daemon=True).start()
    time.sleep(1)
    
    print(f"\nServer ready! Ctrl+C to stop.")
    print(f"To push instructions from another terminal:")
    print(f"  python instruct.py <action> [params_json]")
    print(f"  Example: python instruct.py retry '{{\"n_trees\": 3000}}'\n")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        cleanup()
