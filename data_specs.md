# Especificaciones del Dataset de Startups

## Descripción General
Dataset que contiene información de aproximadamente 923 startups de EE.UU. fundadas entre 2005 y 2013, incluyendo características variadas sobre cada startup.

## Estructura de Datos

### Variables de Identificación
- `id`: Identificador único de la startup
- `name`: Nombre de la startup
- `object_id`: ID alternativo

### Variables Geográficas
- `state_code`: Código del estado
- `city`: Ciudad
- `zip_code`: Código postal
- `latitude`: Latitud
- `longitude`: Longitud
- Variables binarias de estado:
  - `is_CA`: California
  - `is_NY`: Nueva York
  - `is_MA`: Massachusetts
  - `is_TX`: Texas
  - `is_otherstate`: Otros estados

### Variables Temporales
- `founded_at`: Fecha de fundación
- `closed_at`: Fecha de cierre (si aplica)
- `first_funding_at`: Fecha del primer financiamiento
- `last_funding_at`: Fecha del último financiamiento
- `age_first_funding_year`: Edad en años al primer financiamiento
- `age_last_funding_year`: Edad en años al último financiamiento
- `age_first_milestone_year`: Edad en años al primer hito
- `age_last_milestone_year`: Edad en años al último hito

### Variables de Financiamiento
- `funding_rounds`: Número de rondas de financiamiento
- `funding_total_usd`: Total de financiamiento en USD
- `has_VC`: Si tiene capital de riesgo
- `has_angel`: Si tiene inversores ángeles
- `has_roundA`: Si tiene ronda A
- `has_roundB`: Si tiene ronda B
- `has_roundC`: Si tiene ronda C
- `has_roundD`: Si tiene ronda D
- `avg_participants`: Promedio de participantes en rondas

### Variables de Categorización
- `category_code`: Código de categoría
- Variables binarias de categoría:
  - `is_software`: Software
  - `is_web`: Web
  - `is_mobile`: Mobile
  - `is_enterprise`: Enterprise
  - `is_advertising`: Advertising
  - `is_gamesvideo`: Games/Video
  - `is_ecommerce`: E-commerce
  - `is_biotech`: Biotech
  - `is_consulting`: Consulting
  - `is_othercategory`: Otras categorías

### Variables de Éxito
- `labels`: Variable objetivo (éxito/fracaso)
- `status`: Estado actual de la startup
- `is_top500`: Si está en el top 500
- `relationships`: Número de relaciones
- `milestones`: Número de hitos alcanzados

## Calidad de Datos

### Valores Faltantes
- `closed_at`: 63.71% faltantes
- `age_first_milestone_year`: 16.47% faltantes
- `age_last_milestone_year`: 16.47% faltantes
- `Unnamed: 6`: 53.41% faltantes
- `state_code.1`: 0.11% faltantes

### Consideraciones
1. Los valores faltantes en `closed_at` pueden indicar startups activas
2. Los valores faltantes en variables de edad pueden requerir imputación
3. La variable `Unnamed: 6` parece ser redundante y podría ser eliminada

## Transformaciones Necesarias
1. Convertir fechas a formato datetime
2. Crear variables derivadas de tiempo
3. Normalizar variables numéricas
4. Codificar variables categóricas
5. Manejar valores faltantes
6. Crear características compuestas 