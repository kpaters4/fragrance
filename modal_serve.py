"""Modal deployment for the Fragrance API.

Deploy with: modal deploy modal_serve.py
"""
import modal

# sklearn pinned to the EXACT version recorded in pipeline.joblib's
# metadata.sklearn_version (see build.py output) to avoid unpickling issues.
SKLEARN_VERSION = "1.7.2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "pydantic",
        "joblib",
        "numpy",
        "scipy",
        "supabase",
        f"scikit-learn=={SKLEARN_VERSION}",
    )
    .add_local_file("serve.py", "/root/serve.py")
    .add_local_file("pipeline_def.py", "/root/pipeline_def.py")
    .add_local_file("db.py", "/root/db.py")
    .add_local_file("pipeline.joblib", "/root/pipeline.joblib")
)

app = modal.App("scent-fingerprint", image=image)


@app.function(secrets=[modal.Secret.from_name("fragrance-supabase")])
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/root")
    import os

    os.chdir("/root")

    from serve import app as web_app

    return web_app
