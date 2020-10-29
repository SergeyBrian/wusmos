# docker build -t wusmos:latest . && docker run -p 3000:3000 wusmos:latest

FROM python:3.6

COPY ./requirements.txt .

RUN pip install -r requirements.txt
COPY . .


CMD ["python", "app.py"]
