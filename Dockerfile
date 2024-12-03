# Gunakan image dasar Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Salin semua file proyek ke dalam container
COPY . .

# Install dependensi aplikasi
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable untuk service account key
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Expose port aplikasi
EXPOSE 8000

# Jalankan aplikasi FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
