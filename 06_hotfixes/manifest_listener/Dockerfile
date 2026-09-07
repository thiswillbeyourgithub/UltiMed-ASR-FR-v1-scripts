FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir flask==3.0.3
COPY app.py /app/app.py
COPY static /app/static
EXPOSE 5000
CMD ["python", "app.py"]
