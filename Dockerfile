
# Base Python image (slim version to be lighter)
FROM python:3.12-slim


# Install uv by copying the binary from the official uv image (official recommended approach)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Work inside /app
WORKDIR /app

# Copy project metadata
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --locked --no-cache


# Copy full project
COPY . .

# Expose API port
EXPOSE 9696


# Run the FastAPI app with uvicorn via uv
CMD ["uv", "run", "uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "9696"]
