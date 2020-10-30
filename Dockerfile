# Build: docker build -t wusmos:latest .
# Run: docker run -p 3000:3000 -e PORT=3000 wusmos:latest

FROM python:3.6

#RUN apk add --no-cache --update python3 py3-pip bash
ADD ./webapp/requirements.txt /tmp/requirements.txt

#RUN pip3 install pip==19.3.1 setuptools==41.2.0
RUN pip3 install wheel
RUN pip3 install --no-cache-dir -q -r /tmp/requirements.txt

ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

#RUN adduser -D myuser
#USER myuser

CMD gunicorn --bind 0.0.0.0:$PORT wsgi