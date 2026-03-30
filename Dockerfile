# -------- minimal, production-grade runtime --------
FROM python:3.11-slim

# -------- environment settings --------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------- working directory --------
WORKDIR /app

# -------- install dependencies (cached layer) --------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------- copy only required source code --------
COPY train.py .
COPY generate_data.py .

# -------- create runtime directories --------
RUN mkdir -p /app/model /app/data

# -------- default execution --------
CMD ["python", "train.py"]