# Thesis Evolution Benchmark

**Платформа эволюционного бенчмарка для оптимизации генерации дипломных работ (ВКР)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📑 Оглавление

- [Описание проекта](#описание-проекта)
- [Основные возможности](#основные-возможности)
- [Установка и настройка](#установка-и-настройка)
- [Быстрый старт](#быстрый-старт)
- [Режимы работы](#режимы-работы)
- [Конфигурационные файлы](#конфигурационные-файлы)
- [Структура проекта](#структура-проекта)
- [Интерпретация результатов](#интерпретация-результатов)
- [Как работает эволюция](#как-работает-эволюция)
- [Типичные сценарии использования](#типичные-сценарии-использования)
- [Технические требования](#технические-требования)
- [Архитектура системы](#архитектура-системы)

---

## Описание проекта

**Thesis Evolution Benchmark** — это доменно-агностическая платформа для эволюционной оптимизации промптов и структур пайплайнов генерации дипломных работ. Система использует генетические алгоритмы и мета-промпты для автоматического улучшения качества генерации текста через множественные поколения кандидатов.

### Концепция

Проект реализует два взаимодополняющих режима эволюции:

1. **Режим 1: Эволюция промптов** — оптимизация текстов промптов при фиксированной структуре пайплайна
2. **Режим 2: Эволюция пайплайнов** — оптимизация структуры пайплайна при использовании оптимизированных промптов

Оба режима используют систему оценки с тремя типами оценщиков и мета-промпты для генерации улучшенных вариантов.

---

## Основные возможности

### 🎯 Режим 1: Эволюция промптов (Mode 1)

**Цель:** Найти оптимальные промпты для каждого этапа фиксированного пайплайна.

**Процесс:**
- Создаётся популяция промптов для каждого этапа (semantic, outline, intro, chapter1, review)
- Каждый промпт оценивается на множестве тем
- Отбираются топ-k промптов по агрегированной оценке
- Мета-промпты генерируют улучшенные варианты
- Применяются мутации для исследования пространства решений
- Процесс повторяется для заданного числа поколений

**Результат:** Набор оптимизированных промптов для каждого этапа пайплайна.

### 🔄 Режим 2: Эволюция пайплайнов (Mode 2)

**Цель:** Найти оптимальную структуру пайплайна при использовании лучших промптов.

**Процесс:**
- Создаётся начальная популяция пайплайнов (10-20 вариантов)
- Каждый пайплайн оценивается на множестве тем
- Отбираются топ-k пайплайнов
- Генерируются новые варианты через:
  - Мета-генерацию (создание новых структур)
  - Мутацию (изменение температуры, порядка этапов, включение/отключение этапов)
  - Кроссовер (комбинирование лучших пайплайнов)
- Процесс повторяется с детекцией дрифта производительности

**Результат:** Оптимальная структура пайплайна с указанием лучших промптов.

### 🧪 Одиночный прогон пайплайна

**Цель:** Протестировать конкретный пайплайн на одной теме.

**Использование:**
- Быстрая проверка конфигурации
- Отладка промптов
- Валидация результатов перед эволюцией

### 📊 Система оценки

Система использует три типа оценщиков:

| Оценщик | Тип | Описание |
|---------|-----|----------|
| **RuleBasedEvaluator** | Правила | Эвристические метрики: длина текста, структура, полнота |
| **LLMJudgeEvaluator** | LLM-судья | Оценка качества через промпты (общее качество, риск галлюцинаций) |
| **ConsistencyEvaluator** | Консистентность | Проверка согласованности между этапами |

Каждый этап оценивается всеми тремя оценщиками, результаты агрегируются в `aggregated_score`.

### 📁 Структура данных

Все результаты сохраняются в каталоге `data/`:

```
data/
├── runs/              # Сводки прогонов (summary.json)
├── outputs/           # Выходы каждого этапа (stage_type.json)
├── evaluations/       # Детальные оценки (evaluations.json)
├── generations/       # Поколения эволюции
│   ├── prompt_mode/   # Поколения промптов
│   └── pipeline_mode/ # Поколения пайплайнов
└── reports/           # Финальные отчёты (CSV, JSON)
```

---

## Установка и настройка

### Требования

- **Python 3.10+**
- **macOS или Linux** (Windows не тестировался)
- **API ключ DeepSeek** (модель `deepseek-reasoner`)

### Пошаговая установка

#### 1. Клонирование репозитория

```bash
git clone https://github.com/pavelaleks/thesis-evo-bench.git
cd thesis-evo-bench
```

#### 2. Создание виртуального окружения

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Проверка активации:**

```bash
which python  # Должен показать путь к venv/bin/python
```

#### 3. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Или через pyproject.toml:**

```bash
pip install -e .
```

#### 4. Настройка переменных окружения

```bash
cp .env.example .env
```

Отредактируйте `.env` файл:

```env
LLM_API_KEY=your_deepseek_api_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
DEFAULT_MODEL=deepseek-reasoner
```

> **Важно:** Получите API ключ на [DeepSeek Platform](https://platform.deepseek.com/). Модель `deepseek-reasoner` поддерживает до 64K токенов вывода и chain-of-thought reasoning.

#### 5. Проверка установки

Запустите тестовый прогон:

```bash
python -m thesis_evo_bench.cli.main run-single-pipeline \
    --pipeline configs/pipeline_fixed.yaml \
    --topic "Тестовая тема" \
    --domain "Computer Science"
```

Если всё настроено правильно, вы увидите:
- Логи выполнения этапов
- Сохранённые файлы в `data/outputs/` и `data/runs/`
- Финальную оценку в консоли

---

## Быстрый старт

### Пример 1: Одиночный прогон

```bash
python -m thesis_evo_bench.cli.main run-single-pipeline \
    --pipeline configs/pipeline_fixed.yaml \
    --topic "Машинное обучение в медицине" \
    --domain "Computer Science / Healthcare" \
    --description "Исследование применения ML для диагностики"
```

**Что происходит:**
1. Загружается пайплайн из `configs/pipeline_fixed.yaml`
2. Выполняются все этапы последовательно (semantic → outline → intro → chapter1 → review)
3. Каждый этап оценивается тремя оценщиками
4. Результаты сохраняются в `data/runs/{run_id}_summary.json`

### Пример 2: Эволюция промптов (быстрый режим)

Создайте конфиг для быстрого теста (`configs/quick_test.yaml`):

```yaml
generations: 2
population_size_per_stage: 5
selection_top_k: 2
mutation_rate: 0.3
random_seed: 42
```

Запуск:

```bash
python -m thesis_evo_bench.cli.main run-prompt-evolution \
    --config configs/quick_test.yaml \
    --topics configs/topics_example.yaml \
    --pipeline configs/pipeline_fixed.yaml
```

### Пример 3: Эволюция пайплайнов

```bash
python -m thesis_evo_bench.cli.main run-pipeline-evolution \
    --config configs/evolution_pipeline_config.yaml \
    --topics configs/topics_example.yaml \
    --initial-population configs/pipelines_population_example.yaml
```

---

## Режимы работы

### Режим 1: Эволюция промптов

**Команда:**

```bash
python -m thesis_evo_bench.cli.main run-prompt-evolution \
    --config <config_file> \
    --topics <topics_file> \
    --pipeline <pipeline_file>
```

**Параметры:**

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--config` | YAML-файл с настройками эволюции | `configs/evolution_prompt_config.yaml` |
| `--topics` | YAML-файл со списком тем | `configs/topics_example.yaml` |
| `--pipeline` | YAML-файл с фиксированным пайплайном | `configs/pipeline_fixed.yaml` |

**Процесс:**

1. **Инициализация:** Для каждого этапа создаётся популяция промптов (базовый + варианты)
2. **Оценка:** Каждый промпт тестируется на всех темах из файла
3. **Отбор:** Выбираются топ-k промптов по агрегированной оценке
4. **Генерация:** Мета-промпты создают улучшенные варианты
5. **Мутация:** Применяются случайные мутации для разнообразия
6. **Сохранение:** Каждое поколение сохраняется в `data/generations/prompt_mode/generation_N/`

**Результаты:**

- `data/generations/prompt_mode/generation_N/generation_N.yaml` — метаданные поколения
- `data/reports/prompt_evolution/prompt_evolution_report.json` — финальный отчёт
- `data/reports/prompt_evolution/prompt_evolution_scores.csv` — матрица оценок по этапам

### Режим 2: Эволюция пайплайнов

**Команда:**

```bash
python -m thesis_evo_bench.cli.main run-pipeline-evolution \
    --config <config_file> \
    --topics <topics_file> \
    --initial-population <pipelines_file>
```

**Параметры:**

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--config` | YAML-файл с настройками эволюции | `configs/evolution_pipeline_config.yaml` |
| `--topics` | YAML-файл со списком тем | `configs/topics_example.yaml` |
| `--initial-population` | YAML-файл с начальной популяцией пайплайнов | `configs/pipelines_population_example.yaml` |

**Процесс:**

1. **Инициализация:** Загружается начальная популяция пайплайнов (10-20 вариантов)
2. **Оценка:** Каждый пайплайн тестируется на всех темах
3. **Отбор:** Выбираются топ-k пайплайнов
4. **Генерация новых:**
   - **Мета-генерация:** LLM создаёт новые структуры на основе анализа лучших
   - **Мутация:** Изменение температуры, порядка этапов, включение/отключение этапов
   - **Кроссовер:** Комбинирование этапов из двух лучших пайплайнов
5. **Детекция дрифта:** Проверка деградации производительности между поколениями
6. **Сохранение:** Каждый пайплайн сохраняется в `data/generations/pipeline_mode/generation_N/`

**Результаты:**

- `data/generations/pipeline_mode/generation_N/{pipeline_id}.yaml` — конфигурации пайплайнов
- `data/generations/pipeline_mode/generation_N/generation_summary.yaml` — сводка поколения
- `data/reports/pipeline_evolution/pipeline_evolution_report.json` — финальный отчёт
- `data/reports/pipeline_evolution/pipeline_evolution_scores.csv` — тренды оценок

### Одиночный прогон

**Команда:**

```bash
python -m thesis_evo_bench.cli.main run-single-pipeline \
    --pipeline <pipeline_file> \
    --topic "<topic_title>" \
    [--domain "<domain>"] \
    [--description "<description>"]
```

**Параметры:**

| Параметр | Обязательный | Описание |
|----------|--------------|----------|
| `--pipeline` | Да | YAML-файл с конфигурацией пайплайна |
| `--topic` | Да | Название темы |
| `--domain` | Нет | Домен темы (по умолчанию: "general") |
| `--description` | Нет | Описание темы |

**Пример:**

```bash
python -m thesis_evo_bench.cli.main run-single-pipeline \
    --pipeline configs/pipeline_fixed.yaml \
    --topic "Квантовые вычисления в криптографии" \
    --domain "Computer Science / Mathematics" \
    --description "Исследование влияния квантовых компьютеров на криптографическую безопасность"
```

---

## Конфигурационные файлы

### `configs/topics_example.yaml`

Список тем для тестирования. Каждая тема содержит:

```yaml
- title: "Название темы"
  domain: "Домен / Поддомен"
  description: "Подробное описание темы"
```

**Пример:**

```yaml
- title: "Машинное обучение в диагностике здоровья"
  domain: "Computer Science / Healthcare"
  description: "Исследование применения алгоритмов машинного обучения для медицинской диагностики и оптимизации ухода за пациентами"
```

### `configs/pipeline_fixed.yaml`

Конфигурация фиксированного пайплайна для Mode 1. Содержит:

```yaml
pipeline_id: "baseline_pipeline"
name: "Базовый пайплайн генерации диплома"
description: "Стандартный 5-этапный пайплайн"

stages:
  - stage_type: "semantic"
    prompt_profile_name: "semantic_v1"
    temperature: 0.7
    max_tokens: 2000
    enabled: true
    depends_on: []
  
  - stage_type: "outline"
    prompt_profile_name: "outline_v1"
    temperature: 0.7
    max_tokens: 3000
    enabled: true
    depends_on: ["semantic"]
  
  # ... остальные этапы
```

**Параметры этапа:**

| Параметр | Описание | Пример |
|----------|----------|--------|
| `stage_type` | Тип этапа | `semantic`, `outline`, `intro`, `chapter1`, `review` |
| `prompt_profile_name` | Имя промпт-профиля | `semantic_v1`, `outline_v1` |
| `temperature` | Температура генерации (0.0-2.0) | `0.7` |
| `max_tokens` | Максимум токенов вывода | `2000`, `4000` |
| `enabled` | Включён ли этап | `true`, `false` |
| `depends_on` | Зависимости от других этапов | `["semantic"]` |

### `configs/pipelines_population_example.yaml`

Начальная популяция пайплайнов для Mode 2. Список конфигураций пайплайнов:

```yaml
- pipeline_id: "pipeline_001"
  name: "Стандартный последовательный пайплайн"
  stages:
    - stage_type: "semantic"
      prompt_profile_name: "semantic_v1"
      temperature: 0.7
      enabled: true
    # ... остальные этапы
```

### `configs/evolution_prompt_config.yaml`

Настройки эволюции промптов (Mode 1):

```yaml
generations: 5                    # Количество поколений
population_size_per_stage: 10    # Размер популяции на этап
selection_top_k: 3               # Топ-k для отбора
mutation_rate: 0.3                # Вероятность мутации
random_seed: 42                   # Seed для воспроизводимости
```

**Параметры:**

| Параметр | Описание | Рекомендации |
|----------|----------|--------------|
| `generations` | Число поколений | 5-10 для исследования, 2-3 для быстрого теста |
| `population_size_per_stage` | Промптов на этап | 10-20 для качества, 5 для скорости |
| `selection_top_k` | Отбираемых лучших | 20-30% от популяции |
| `mutation_rate` | Вероятность мутации | 0.2-0.4 |
| `random_seed` | Seed для RNG | Любое число для воспроизводимости |

### `configs/evolution_pipeline_config.yaml`

Настройки эволюции пайплайнов (Mode 2):

```yaml
generations: 5
population_size: 15
selection_top_k: 5
mutation_rate: 0.3
crossover_rate: 0.4
random_seed: 42

pipeline_constraints:
  min_stages: 2
  max_stages: 10
  allowed_stage_types:
    - "semantic"
    - "outline"
    - "intro"
    - "chapter1"
    - "review"
```

**Параметры:**

| Параметр | Описание | Рекомендации |
|----------|----------|--------------|
| `population_size` | Размер популяции | 15-20 для качества, 8-10 для скорости |
| `crossover_rate` | Вероятность кроссовера | 0.3-0.5 |
| `min_stages` | Минимум этапов | 2-3 |
| `max_stages` | Максимум этапов | 8-10 |

---

## Структура проекта

```
thesis-evo-bench/
├── thesis_evo_bench/          # Основной пакет
│   ├── config/                # Управление конфигурацией
│   │   └── settings.py        # Загрузка переменных окружения
│   ├── llm/                   # Клиент LLM
│   │   └── client.py          # HTTP-обёртка для DeepSeek API
│   ├── core/                  # Ядро системы эволюции
│   │   ├── models.py          # Pydantic-модели (Topic, Pipeline, Stage, etc.)
│   │   ├── prompts.py         # Реестр 25 промпт-профилей
│   │   ├── evaluators.py      # Три типа оценщиков
│   │   ├── pipeline_executor.py # Исполнитель пайплайнов
│   │   ├── evolution_prompt_mode.py    # Режим 1: эволюция промптов
│   │   ├── evolution_pipeline_mode.py  # Режим 2: эволюция пайплайнов
│   │   ├── reporting.py       # Генерация отчётов (CSV, JSON)
│   │   └── utils.py           # Утилиты (загрузка YAML, рендеринг промптов)
│   └── cli/                   # CLI-интерфейс
│       └── main.py             # Три команды (run-single-pipeline, run-prompt-evolution, run-pipeline-evolution)
├── configs/                    # YAML-конфигурации
│   ├── topics_example.yaml
│   ├── pipeline_fixed.yaml
│   ├── pipelines_population_example.yaml
│   ├── evolution_prompt_config.yaml
│   └── evolution_pipeline_config.yaml
├── data/                       # Результаты работы
│   ├── runs/                   # Сводки прогонов
│   ├── outputs/                # Выходы этапов
│   ├── evaluations/           # Детальные оценки
│   ├── generations/            # Поколения эволюции
│   │   ├── prompt_mode/
│   │   └── pipeline_mode/
│   └── reports/                # Финальные отчёты
├── pyproject.toml              # Конфигурация проекта
├── requirements.txt            # Зависимости
├── .env.example                # Шаблон переменных окружения
└── README.md                   # Этот файл
```

### Модули

#### `thesis_evo_bench/core/prompts.py`

Содержит **25 промпт-профилей**, разделённых на категории:

1. **Генеративные промпты (5):**
   - `semantic_v1` — семантический анализ темы
   - `outline_v1` — генерация структуры диплома
   - `intro_v1` — написание введения
   - `chapter1_v1` — написание первой главы
   - `review_v1` — рецензирование и предложения по улучшению

2. **Оценочные промпты (4):**
   - `eval_general_quality_v1` — общая оценка качества
   - `eval_hallucination_risk_v1` — оценка риска галлюцинаций
   - `eval_structure_coherence_v1` — оценка структурной связности
   - `eval_cross_consistency_v1` — оценка кросс-консистентности

3. **Мета-промпты для эволюции промптов (8):**
   - `meta_improve_*_prompt_v1` — улучшение промптов по этапам
   - `meta_analyze_prompt_weaknesses_v1` — анализ слабостей
   - `meta_generate_prompt_mutation_v1` — генерация мутаций
   - `meta_compare_prompts_v1` — сравнение вариантов
   - `meta_select_best_prompt_v1` — выбор лучшего

4. **Мета-промпты для эволюции пайплайнов (5):**
   - `meta_analyze_pipeline_v1` — анализ пайплайна
   - `meta_generate_new_pipeline_v1` — генерация новых структур
   - `meta_mutate_pipeline_v1` — мутация пайплайнов
   - `meta_crossover_pipeline_v1` — кроссовер пайплайнов
   - `meta_check_pipeline_drift_v1` — проверка дрифта

5. **Промпты управления эволюцией (3):**
   - `meta_evolution_summary_v1` — сводка прогресса
   - `meta_regression_check_v1` — проверка регрессии
   - `meta_generate_experiment_report_v1` — генерация отчёта

---

## Интерпретация результатов

### Структура `summary.json`

Файл `data/runs/{run_id}_summary.json` содержит полную сводку прогона:

```json
{
  "run_id": "2bdc262b-4fc9-4c10-afbd-b724d2c43d0e",
  "pipeline_id": "baseline_pipeline",
  "topic": {
    "title": "Машинное обучение в медицине",
    "domain": "Computer Science / Healthcare"
  },
  "stage_results": [
    {
      "stage_type": "semantic",
      "output_content": "...",
      "token_usage": {
        "prompt_tokens": 150,
        "completion_tokens": 500,
        "total_tokens": 650
      },
      "execution_time_seconds": 3.2
    }
    // ... остальные этапы
  ],
  "evaluations": [
    {
      "evaluator_name": "rule_based",
      "evaluator_type": "rule_based",
      "score": 0.75,
      "reasoning": "Rule-based evaluation: length=0.8, structure=0.7, completeness=0.75"
    }
    // ... остальные оценки
  ],
  "aggregated_score": 0.72,
  "execution_time_seconds": 45.3
}
```

**Ключевые поля:**

- `aggregated_score` — средняя оценка по всем оценщикам (0.0-1.0, выше = лучше)
- `stage_results` — результаты каждого этапа с токенами и временем
- `evaluations` — детальные оценки с reasoning

### Каталог `outputs/`

Содержит JSON-файлы выходов каждого этапа:

- `{run_id}_semantic.json` — семантический анализ
- `{run_id}_outline.json` — структура диплома
- `{run_id}_intro.json` — введение
- `{run_id}_chapter1.json` — первая глава
- `{run_id}_review.json` — рецензия

Каждый файл содержит полный `StageRunResult` с метаданными.

### Каталог `evaluations/`

Содержит `{run_id}_evaluations.json` с детальными оценками:

```json
[
  {
    "evaluator_name": "llm_judge",
    "score": 0.82,
    "reasoning": "Content demonstrates high clarity and coherence...",
    "details": {
      "clarity": 0.85,
      "coherence": 0.80,
      "completeness": 0.81
    }
  }
]
```

### Каталог `reports/`

#### `prompt_evolution_report.json`

```json
{
  "evolution_id": "...",
  "total_generations": 5,
  "final_best_prompts": {
    "semantic": {
      "name": "semantic_v1_gen5_improved2",
      "template": "..."
    }
  },
  "stage_scores": {
    "semantic": [0.65, 0.72, 0.78, 0.81, 0.83]
  }
}
```

#### `prompt_evolution_scores.csv`

CSV-таблица с оценками по поколениям и этапам:

| generation | semantic | outline | intro | chapter1 | review |
|------------|----------|---------|-------|----------|--------|
| 0 | 0.65 | 0.70 | 0.68 | 0.72 | 0.71 |
| 1 | 0.72 | 0.75 | 0.73 | 0.76 | 0.74 |
| ... | ... | ... | ... | ... | ... |

#### `pipeline_evolution_report.json`

```json
{
  "evolution_id": "...",
  "total_generations": 5,
  "final_best_pipeline": {
    "pipeline_id": "pipeline_gen5_cross3",
    "final_score": 0.85,
    "stages": [...]
  },
  "generation_scores": [
    {"generation": 0, "best_score": 0.70, "average_score": 0.65},
    {"generation": 1, "best_score": 0.75, "average_score": 0.71}
  ]
}
```

### Как читать оценки

**`aggregated_score` (0.0-1.0):**

- **0.9-1.0:** Отличное качество, готово к использованию
- **0.7-0.9:** Хорошее качество, возможны небольшие улучшения
- **0.5-0.7:** Среднее качество, требуется доработка
- **0.0-0.5:** Низкое качество, нужна значительная оптимизация

**`reasoning` в evaluations:**

Содержит объяснение оценки от LLM-судьи или детали от rule-based оценщика. Используйте для понимания сильных и слабых сторон.

**Ошибки в `metadata`:**

Если этап завершился с ошибкой, в `metadata` будет поле `"error"` с описанием. Проверьте логи для деталей.

---

## Как работает эволюция

### Генерация популяции

**Mode 1 (Промпты):**

1. Базовый промпт копируется N раз
2. К каждой копии применяются случайные мутации:
   - Добавление инструкций
   - Изменение формулировок
   - Модификация структуры вывода
3. Создаётся начальная популяция размером `population_size_per_stage`

**Mode 2 (Пайплайны):**

1. Загружается начальная популяция из YAML
2. Если популяция меньше `population_size`, генерируются дополнительные через мутацию существующих

### Мутация

**Промпты:**

- Замена фраз ("JSON format" → "structured JSON format")
- Добавление инструкций
- Изменение требований к выводу

**Пайплайны:**

- Изменение `temperature` (±0.2)
- Включение/отключение этапов
- Добавление новых этапов
- Изменение порядка этапов

### Кроссовер (только для пайплайнов)

1. Выбираются два родительских пайплайна (топ-k)
2. Комбинируются этапы из обоих
3. Создаётся потомок с лучшими характеристиками обоих родителей

### Критерии отбора

1. **Оценка на множестве тем:** Каждый кандидат тестируется на всех темах из `topics_example.yaml`
2. **Агрегирование:** Средняя оценка по всем темам и всем оценщикам
3. **Ранжирование:** Сортировка по убыванию `aggregated_score`
4. **Отбор:** Выбираются топ-k кандидатов

### Сохранение поколений

**Mode 1:**

```
data/generations/prompt_mode/
├── generation_0/
│   └── generation_0.yaml        # Метаданные: популяция, оценки, лучшие
├── generation_1/
│   └── generation_1.yaml
└── ...
```

**Mode 2:**

```
data/generations/pipeline_mode/
├── generation_0/
│   ├── pipeline_gen0_0.yaml
│   ├── pipeline_gen0_1.yaml
│   ├── ...
│   └── generation_summary.yaml
└── ...
```

---

## Типичные сценарии использования

### Сценарий 1: Исследовательский режим (долгий)

**Цель:** Найти оптимальные промпты для продакшн-использования.

**Настройки:**

```yaml
# evolution_prompt_config.yaml
generations: 10
population_size_per_stage: 20
selection_top_k: 5
mutation_rate: 0.3
```

**Время выполнения:** 2-4 часа (зависит от числа тем и размера популяции)

**Ожидаемый результат:** Промпты с оценкой 0.85+ для всех этапов

### Сценарий 2: Быстрый режим

**Цель:** Быстро протестировать идеи или отладить конфигурацию.

**Настройки:**

```yaml
# quick_test.yaml
generations: 2
population_size_per_stage: 5
selection_top_k: 2
mutation_rate: 0.3
```

**Время выполнения:** 15-30 минут

**Ожидаемый результат:** Базовое понимание направления оптимизации

### Сценарий 3: Глубокая эволюция

**Цель:** Максимально оптимизировать систему через комбинирование Mode 1 и Mode 2.

**Шаг 1:** Эволюция промптов (Mode 1)

```bash
python -m thesis_evo_bench.cli.main run-prompt-evolution \
    --config configs/evolution_prompt_config.yaml \
    --topics configs/topics_example.yaml \
    --pipeline configs/pipeline_fixed.yaml
```

**Шаг 2:** Использование лучших промптов в пайплайнах

Обновите `pipelines_population_example.yaml`, заменив `prompt_profile_name` на лучшие из Step 1.

**Шаг 3:** Эволюция пайплайнов (Mode 2)

```bash
python -m thesis_evo_bench.cli.main run-pipeline-evolution \
    --config configs/evolution_pipeline_config.yaml \
    --topics configs/topics_example.yaml \
    --initial-population configs/pipelines_population_example.yaml
```

**Результат:** Оптимизированная система с лучшими промптами и структурой пайплайна

---

## Технические требования

### Системные требования

- **ОС:** macOS 10.15+, Linux (Ubuntu 20.04+, Debian 11+)
- **Python:** 3.10, 3.11, 3.12
- **RAM:** Минимум 4 GB, рекомендуется 8+ GB
- **Диск:** 500 MB для установки, +1-5 GB для данных

### Зависимости

Все зависимости указаны в `requirements.txt`:

```
pydantic>=2.0.0
pyyaml>=6.0
pandas>=2.0.0
tqdm>=4.65.0
httpx>=0.24.0
python-dotenv>=1.0.0
```

### DeepSeek API

**Модель:** `deepseek-reasoner`

**Характеристики:**
- До 64K токенов вывода
- Chain-of-thought reasoning
- Стабильный JSON-вывод
- Улучшенное суждение для оценки и мутации

**Лимиты токенов:**

| Параметр | Значение |
|----------|----------|
| Максимум входных токенов | 128K |
| Максимум выходных токенов | 64K |
| Рекомендуемый `max_tokens` для этапов | 2000-4000 |

**Ориентировочная стоимость:**

Примерная стоимость одного прогона пайплайна (5 этапов):

- **Входные токены:** ~2000-3000 (промпты + контекст)
- **Выходные токены:** ~8000-12000 (генерация текста)
- **Стоимость:** ~$0.01-0.02 за прогон (зависит от тарифа DeepSeek)

**Эволюция промптов (5 поколений, 10 промптов на этап, 5 тем):**

- **Всего прогонов:** ~1250 (5 этапов × 10 промптов × 5 тем × 5 поколений)
- **Стоимость:** ~$12-25 за полную эволюцию

> **Совет:** Начните с быстрого режима (2 поколения, 5 промптов) для оценки стоимости.

---

## Архитектура системы

### Компоненты

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Interface                         │
│  (run-single-pipeline, run-prompt-evolution, etc.)      │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐            ┌─────▼─────┐
    │  Mode 1 │            │   Mode 2   │
    │ Prompts │            │ Pipelines │
    └────┬────┘            └─────┬─────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │  Pipeline Executor     │
         │  (run_pipeline_for_topic)│
         └───────────┬───────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼───┐      ┌─────▼─────┐   ┌─────▼─────┐
│ LLM   │      │Evaluators │   │  Prompts  │
│Client │      │(3 types)  │   │ Registry  │
└───┬───┘      └───────────┘   └───────────┘
    │
┌───▼──────────────┐
│  DeepSeek API    │
│  (deepseek-      │
│   reasoner)      │
└──────────────────┘
```

### Поток данных

1. **CLI** загружает конфигурации (YAML)
2. **Evolution Engine** создаёт популяцию кандидатов
3. **Pipeline Executor** выполняет пайплайн для каждой темы
4. **LLM Client** отправляет запросы к DeepSeek API
5. **Evaluators** оценивают результаты
6. **Reporting** генерирует отчёты и сохраняет в `data/`

### Безопасность рендеринга промптов

Система использует безопасный рендеринг через `render_prompt()`:

- Отсутствующие переменные заменяются на пустые строки (не вызывают ошибок)
- Контекст накапливается между этапами
- При ошибке этапа последующие этапы продолжают работу с пустыми значениями

### Обработка ошибок

- **LLM API ошибки:** Повторные попытки с экспоненциальной задержкой (3 попытки)
- **Ошибки этапов:** Логируются, этап помечается как failed, выполнение продолжается
- **Ошибки оценки:** Fallback на rule-based оценщик

---

## Полезные советы

### Оптимизация производительности

1. **Используйте меньше тем для быстрых тестов:** Создайте `topics_quick.yaml` с 2-3 темами
2. **Уменьшите `max_tokens`:** Для тестирования достаточно 1000-2000 токенов на этап
3. **Параллелизация:** Система последовательная, но можно запускать несколько процессов для разных конфигураций

### Отладка

1. **Проверьте логи:** Все ошибки логируются с уровнем WARNING или ERROR
2. **Проверьте `summary.json`:** Поле `metadata.error` содержит описание ошибок
3. **Тестируйте промпты вручную:** Используйте `run-single-pipeline` для отладки

### Воспроизводимость

- Используйте `random_seed` в конфигах для воспроизводимости результатов
- Сохраняйте конфигурации вместе с результатами

---

## Лицензия

MIT License

## Автор

Pavel Alekseev

## Ссылки

- [Репозиторий на GitHub](https://github.com/pavelaleks/thesis-evo-bench)
- [DeepSeek Platform](https://platform.deepseek.com/)

---

**Версия документа:** 1.0  
**Последнее обновление:** 2025
