# TestRapidsAI

Proyecto para ejecutar modelos de machine learning con RAPIDS (cuML) en Docker.

## Requisitos Previos

- Docker y Docker Compose instalados
- NVIDIA Docker runtime (nvidia-docker2) para soporte de GPU
- Una GPU compatible con CUDA

## Cómo Ejecutar el Proyecto

### 1. Construir y ejecutar el contenedor

```bash
docker-compose up --build
```

Este comando:

- Construye la imagen Docker con RAPIDS y todas las dependencias
- Inicia el contenedor con soporte para GPU
- Inicia Jupyter Lab automáticamente

### 2. Acceder a Jupyter Lab

Una vez que el contenedor esté corriendo, abre tu navegador y ve a:

```
http://localhost:8888
```

Jupyter Lab estará disponible sin necesidad de token o contraseña.

### 3. Ejecutar el Notebook

1. Navega a `20_examen/1_models_gpu.ipynb` en Jupyter Lab
2. Abre el notebook
3. Ejecuta las celdas en orden

### 4. Detener el contenedor

Presiona `Ctrl+C` en la terminal donde está corriendo docker-compose, o ejecuta:

```bash
docker-compose down
```

## Estructura del Proyecto

```
workspace/
├── 20_examen/
│   ├── 1_models_gpu.ipynb    # Notebook principal con modelos GPU
│   ├── data/                  # Datos del proyecto
│   └── utils/                 # Utilidades (encoders, scalers, modelos)
├── dockerfile                 # Configuración de la imagen Docker
└── requirements.txt           # Dependencias de Python
```

## Notas Importantes

- El notebook usa modelos de cuML (RAPIDS) que requieren GPU
- Los datos deben estar en `workspace/20_examen/data/`
- El contenedor mapea el directorio `./workspace` al `/workspace` del contenedor
- Los cambios en los archivos se reflejan inmediatamente (volumen montado)

## Solución de Problemas

### El contenedor no inicia

- Verifica que Docker tenga acceso a la GPU: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
- Asegúrate de tener nvidia-docker2 instalado

### Jupyter no es accesible

- Verifica que el puerto 8888 no esté en uso
- Revisa los logs: `docker-compose logs rapidsai`

### Error al importar cuML

- Verifica que la GPU sea compatible con CUDA 13
- Revisa que el contenedor tenga acceso a la GPU: `docker exec rapidsai_custom_container nvidia-smi`
