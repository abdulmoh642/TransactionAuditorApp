FROM python:3.11-slim

# Set environment variables for non-interactive commands and API key hint
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_API_KEY=""
ENV APP_HOME=/app

# Install dependencies needed for document loading (Excel, PDF, etc.)
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libxslt1-dev \
    libz-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    gcc \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR $APP_HOME

# Copy only the dependency file first to cache the layer
COPY requirements.txt .

# Install Python dependencies (including the essential FAISS-CPU for RAG)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install faiss-cpu \
    && pip install openpyxl  # Ensure openpyxl is installed for Excel handling

# Copy the rest of the application code
COPY agent_api.py .
COPY . .

# Expose the port (FastAPI default port)
EXPOSE 80

# Command to run the application using Uvicorn (FastAPI server)
CMD ["uvicorn", "agent_api:app", "--host", "0.0.0.0", "--port", "80"]


