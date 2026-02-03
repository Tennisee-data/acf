"""GitHub Webhook Handler Pattern.

Keywords: github, webhook, signature, hmac, x-hub-signature-256

Similar to Stripe, GitHub webhooks MUST verify signature using raw body.

Requirements:
    pip install fastapi

Security: GitHub uses HMAC-SHA256 for signature verification.
The X-Hub-Signature-256 header contains the signature.
"""

import hmac
import hashlib
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Header
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "your-secret")

app = FastAPI()


def verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature.

    Args:
        payload: Raw request body bytes
        signature: X-Hub-Signature-256 header value
        secret: Your webhook secret

    Returns:
        True if signature is valid
    """
    if not signature.startswith("sha256="):
        return False

    expected_signature = signature[7:]  # Remove "sha256=" prefix

    # Compute HMAC-SHA256
    mac = hmac.new(
        secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256
    )
    computed_signature = mac.hexdigest()

    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(computed_signature, expected_signature)


async def handle_push(payload: dict) -> None:
    """Handle push event."""
    ref = payload.get("ref", "")
    commits = payload.get("commits", [])
    logger.info(f"Push to {ref}: {len(commits)} commits")


async def handle_pull_request(payload: dict) -> None:
    """Handle pull_request event."""
    action = payload.get("action", "")
    pr = payload.get("pull_request", {})
    logger.info(f"PR #{pr.get('number')}: {action}")


async def handle_issues(payload: dict) -> None:
    """Handle issues event."""
    action = payload.get("action", "")
    issue = payload.get("issue", {})
    logger.info(f"Issue #{issue.get('number')}: {action}")


EVENT_HANDLERS = {
    "push": handle_push,
    "pull_request": handle_pull_request,
    "issues": handle_issues,
}


@app.post("/webhooks/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None),
    x_github_delivery: Optional[str] = Header(None),
):
    """GitHub webhook endpoint.

    Headers used:
    - X-Hub-Signature-256: HMAC signature for verification
    - X-GitHub-Event: Event type (push, pull_request, etc.)
    - X-GitHub-Delivery: Unique delivery ID for idempotency
    """
    # Step 1: Get raw body
    payload = await request.body()

    # Step 2: Verify signature
    if not x_hub_signature_256:
        raise HTTPException(status_code=400, detail="Missing signature header")

    if not verify_github_signature(payload, x_hub_signature_256, GITHUB_WEBHOOK_SECRET):
        logger.error("Invalid GitHub webhook signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Step 3: Parse JSON (safe now that signature is verified)
    import json
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Step 4: Log delivery
    logger.info(f"GitHub webhook: {x_github_event} (delivery: {x_github_delivery})")

    # Step 5: Handle event in background
    handler = EVENT_HANDLERS.get(x_github_event)
    if handler:
        background_tasks.add_task(handler, data)
    else:
        logger.info(f"Unhandled event: {x_github_event}")

    return {"status": "received"}
