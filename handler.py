"""RunPod Serverless entrypoint shim.

Docker CMD: python -u /app/handler.py
"""

from api.handler import main

if __name__ == "__main__":
    main()
