# Escolha uma imagem base com Python e outras dependências básicas
FROM python:3.11-slim

# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copie o arquivo de dependências para o container
COPY requirements.txt requirements.txt

# Instale as dependências listadas em requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante dos arquivos do projeto para o container
COPY . .

# Defina o comando para iniciar o script Python
CMD ["python", "train.py"]