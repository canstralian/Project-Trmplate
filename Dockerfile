# --- Builder Stage ---
FROM python:3.10-slim-bullseye AS builder

# Set working directory
WORKDIR /app

# Install system dependencies (required for some packages to build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only dependency files for better layer caching
COPY poetry.lock pyproject.toml ./

# Install dependencies (no virtualenv, no dev deps)
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# --- Final Stage ---
FROM python:3.10-slim-bullseye

# Create and use non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser/app
USER appuser

# Copy only necessary runtime artifacts from builder
COPY --from=builder /app /home/appuser/app

# Copy application source
COPY --chown=appuser:appuser src/ /home/appuser/app/src/

# Expose application port
EXPOSE 8000

# Use exec form CMD (safer with signals, recommended for uvicorn)
CMD ["uvicorn", "src.your_project_name.main:app", "--host", "0.0.0.0", "--port", "8000"]
