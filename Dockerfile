# Imagen base ligera con Python 3.11
FROM python:3.11-slim

# Crear directorios requeridos por SageMaker
RUN mkdir -p /opt/code /opt/model

# Instalar dependencias necesarias
RUN pip install --no-cache-dir pandas scikit-learn fastapi uvicorn pydantic

# Copiar el código fuente al contenedor
COPY extras/local-server/main.py /opt/code/main.py
COPY extras/local-server/model/rf-final-model.pkl /opt/code/model/rf-final-model.pkl
COPY extras/local-server/model/categories-ohe.pkl /opt/code/model/categories-ohe.pkl

# Establecer el directorio de trabajo
WORKDIR /opt/code/

# Crear un alias ejecutable llamado "serve"
RUN echo '#!/bin/sh\npython3 /opt/code/main.py' > /usr/local/bin/serve && chmod +x /usr/local/bin/serve

# SageMaker intentará ejecutar "serve", que lanzará tu FastApi server
ENTRYPOINT ["serve"]