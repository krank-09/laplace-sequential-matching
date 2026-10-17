# Team Laplace policy image. Standard library only: no packages to install.
FROM python:3.12.7-slim
WORKDIR /app
COPY kit.py policy.py ./
COPY laplace/ ./laplace/
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 65534:65534
ENTRYPOINT ["python", "policy.py"]
