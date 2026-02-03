"""Stripe Webhook Handler - Correct Pattern.

Keywords: stripe, webhook, signature, payment, construct_event

CRITICAL: Stripe webhook verification MUST use raw request body bytes.
Using request.json() BEFORE construct_event will ALWAYS fail.

Requirements:
    pip install stripe fastapi

Common mistakes this pattern prevents:
1. Using request.json() before signature verification
2. Not handling idempotency (duplicate events)
3. Blocking on long operations (Stripe has timeout)
4. Not returning 200 quickly enough
"""

import stripe
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import logging

logger = logging.getLogger(__name__)

# Configuration
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "whsec_...")
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "sk_test_...")

app = FastAPI()

# Track processed events for idempotency (use Redis in production)
processed_events: set[str] = set()


async def process_payment_succeeded(event: stripe.Event) -> None:
    """Handle payment_intent.succeeded event.

    This runs in background - don't block the webhook response.
    """
    payment_intent = event.data.object
    logger.info(f"Payment succeeded: {payment_intent.id}")
    # Your business logic here


async def process_payment_failed(event: stripe.Event) -> None:
    """Handle payment_intent.payment_failed event."""
    payment_intent = event.data.object
    logger.warning(f"Payment failed: {payment_intent.id}")
    # Your business logic here


# Event handler mapping
EVENT_HANDLERS = {
    "payment_intent.succeeded": process_payment_succeeded,
    "payment_intent.payment_failed": process_payment_failed,
    # Add more handlers as needed
}


@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    """Stripe webhook endpoint.

    CRITICAL IMPLEMENTATION NOTES:
    1. Use request.body() to get RAW bytes - NOT request.json()
    2. Verify signature BEFORE trusting any data
    3. Check idempotency to handle duplicate deliveries
    4. Return 200 quickly, process in background
    """
    # Step 1: Get RAW body bytes (CRITICAL!)
    payload = await request.body()

    # Step 2: Get signature header
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing signature header")

    # Step 3: Verify signature and construct event
    try:
        event = stripe.Webhook.construct_event(
            payload,      # Raw bytes!
            sig_header,
            STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        # Invalid payload
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Step 4: Idempotency check
    event_id = event.id
    if event_id in processed_events:
        logger.info(f"Duplicate event ignored: {event_id}")
        return {"status": "already_processed"}

    # Mark as processed (before processing to prevent race conditions)
    processed_events.add(event_id)

    # Step 5: Log event for debugging
    logger.info(f"Received event: {event.type} ({event_id})")

    # Step 6: Handle event in background
    handler = EVENT_HANDLERS.get(event.type)
    if handler:
        background_tasks.add_task(handler, event)
    else:
        logger.info(f"Unhandled event type: {event.type}")

    # Step 7: Return 200 immediately
    # Stripe retries on non-2xx or timeout (>20s)
    return {"status": "received"}


# WRONG PATTERN - DO NOT USE:
#
# @app.post("/webhooks/stripe")
# async def wrong_webhook(request: Request):
#     # WRONG: This parses JSON before signature verification
#     data = await request.json()  # <-- BREAKS SIGNATURE!
#
#     payload = await request.body()  # Now payload doesn't match what we parsed
#     event = stripe.Webhook.construct_event(payload, sig, secret)  # FAILS!
