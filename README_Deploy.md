# Deployment

Estou usando o podman para criar a imagem. Basta trocar `podman` para `docker` para usar o docker.

# Docker Hub

Criar o repositório no Docker Hub manualmente.

Criar a imagem:

```bash
podman build --platform=linux/amd64 -t fiap_ia:latest -f Dockerfile
```

Tagear a imagem:

```bash
podman tag localhost/fiap_ia:latest docker.io/smatioli/fiap_ia:latest
```


Após criar a imagem, basta fazer o push para o Docker Hub:

```bash
podman push docker.io/smatioli/fiap_ia:latest
```

Conferir se a imagem está no Docker Hub: https://hub.docker.com/r/smatioli/fiap_ia/tags


# Azure 

O aplicativo será executado em um Azure AppService

Login na Azure:

```bash
az login
```

Criar o Resource Group:

```bash
az group create --name fiap_ps_ia --location eastus
```

```bash
az appservice plan create \
  --name fiap_ps_ia_plan \
  --resource-group fiap_ps_ia \
  --sku B1 \
  --is-linux
```

```bash
az webapp create \
  --resource-group fiap_ps_ia \
  --plan fiap_ps_ia_plan \
  --name fiappsiaapp \
  --deployment-container-image-name docker.io/smatioli/fiap_ia:latest
```

Configurar a porta:

```bash
az webapp config appsettings set \
  --resource-group fiap_ps_ia \
  --name fiappsiaapp \
  --settings WEBSITES_PORT=8000
```

Configurar o health check:

```bash
az webapp config set \
  --resource-group fiap_ps_ia \
  --name fiappsiaapp \
  --generic-configurations '{"healthCheckPath": "/health"}'
```

Iniciar o App Service:

```bash
az webapp start \
  --resource-group fiap_ps_ia \
  --name fiappsiaapp
```

Configuração e streaming de logs:

```bash
az webapp log config --resource-group fiap_ps_ia --name fiappsiaapp --docker-container-logging filesystem
```

```bash
az webapp log tail \
  --resource-group fiap_ps_ia \
  --name fiappsiaapp
```