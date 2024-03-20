FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

# Define environment variable to buffer Python output to make it appear in the logs
ENV PYTHONUNBUFFERED 1

# Run app.py
CMD ["python", "app.py"]
