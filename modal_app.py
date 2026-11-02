"""Deploy the D2 data-audit dashboard (`app/dashboard.py`) on Modal.

    modal secret create supabase-creds SUPABASE_URL=... SUPABASE_KEY=...
    modal deploy modal_app.py

Modal app/function definitions are lazy -- importing this file does not
require a token or network call, only `modal deploy`/`modal run` does. The
Supabase credentials arrive as a Modal secret injected as the same two
environment variable names `src/supabase_client.py` already reads locally,
so there is exactly one credential code path for both places.

The image intentionally does NOT reuse `requirements.txt` wholesale: that
file also pins D1's embedding stack (torch, sentence-transformers, optimum),
which this dashboard never imports, and which has a pre-existing
transformers/sentence-transformers version conflict that a real `pip`
resolve (unlike the sequence of `uv pip install` calls that built the local
`.venv`) refuses outright. Listing only what `app/dashboard.py` + `src/viz.py`
+ `src/supabase_client.py` actually import keeps the image small and avoids
that conflict entirely.
"""

from __future__ import annotations

import shlex
import subprocess

import modal

app = modal.App("dtsc3602-d2-dashboard")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas==3.1.0",
        "matplotlib==3.11.2",
        "streamlit==1.66.1",
        "supabase==2.33.0",
        "python-dotenv==1.2.4",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("app", remote_path="/root/app")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("supabase-creds")],
    min_containers=0,
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=60)
def run() -> None:
    cmd = (
        "streamlit run /root/app/dashboard.py "
        "--server.port 8000 --server.address 0.0.0.0 "
        "--server.headless true --server.enableCORS false"
    )
    subprocess.Popen(shlex.split(cmd))
