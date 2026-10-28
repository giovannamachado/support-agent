FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir . && useradd -m app
USER app
ENV PYTHONUNBUFFERED=1 SUPPORT_AGENT_DATA=/app/data
EXPOSE 8000
CMD ["uvicorn", "support_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
