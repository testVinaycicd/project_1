FROM python:3.11

WORKDIR /app

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your code
COPY train.py .
COPY data/churn_data.csv .

# optional: if you use local modules
# COPY models/ ./models

# default command not required for KFP, but fine to keep
CMD ["python", "train.py"]