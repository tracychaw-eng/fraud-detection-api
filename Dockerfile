FROM python:3.11-slim

WORKDIR /app

# Install dependencies first — Docker layer caching
# means pip install only re-runs when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Copy serve script to PATH and make executable
# SageMaker calls 'serve' to start the container
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

# SageMaker requires port 8080
EXPOSE 8080

# Default command for local testing
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


