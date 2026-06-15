from celery import Celery
import os

# Grab the Redis URL from the environment, defaulting to the Docker network name
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "lead_gen_worker",
    broker=redis_url,
    backend=redis_url,
    # We will tell it exactly where to look for tasks in the next step
    include=['src.agent.tasks']
)

# Standard Celery configurations for JSON serialization
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)