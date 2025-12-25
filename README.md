# LightFM DeepFM Candidate Generator

Этот репозиторий содержит кандидатогенераторную модель рекомендательной системы для b2b- и smb-клиентов. Модель `DeepFM` из `src/models/deepfm.py` обучается на больших потоках транзакций: она строит эмбеддинги для множества плательщиков (dt) и получателей (kt), а затем используется как источник кандидатов перед ранжированием. Поток данных выстраивается вокруг Spark-джоб, которые подготавливают паркетные партиции и словари признаков, а PyTorch-часть обучается на стриминговом датасете `StreamDataset` (`src/datasets/stream_dataset.py`).

## Структура репозитория
- `src/` – основной пакет (датасеты, модели, лоссы, трейн-луп, метрики и конфиги Hydra).
- `notebooks/make_dataset_scripts/` – Spark-ноутбуки и `config.yaml` для выгрузки и агрегации исходных данных.
- `notebooks/process_results/collect_embeddings.ipynb` – выгрузка эмбеддингов и attention-весов обученной модели.
- `notebooks/make_recommendations/` – построение ANN-индекса и офлайн-метрик.
- `train.py` / `inference.py` – Hydra-входные точки для обучения и инференса.

## Поток данных и подготовка
1. **Сбор и агрегация в Spark** (`notebooks/make_dataset_scripts/run.ipynb`, `utils.py`):
   - Конфиг `notebooks/make_dataset_scripts/config.yaml` задаёт параметры Spark-кластера, временные окна train/test и пути в HDFS (эмбеддинги, словари контекстных слов, директории с результатами).
   - Паркетные выгрузки строятся по dt/kt парам, к ним добавляются атрибуты (ОКВЭД, ОКАТО, разбиения БИК/счёта, статистики сумм). Есть отдельные части для контент-фичей (топики LDA по словам из сообщений), генерации хард-негативов и индексации категориальных признаков через `StringIndexer`.
   - На выходе формируются: `...train_interactions/*.parquet`, `...test_interactions/*.parquet`, паспорта dt/kt, маппинги `inn_* -> *_index`, эмбеддинги сторонних моделей и словари уникальных клиентов для негативного сэмплинга.
2. **Упаковка признаков** (`notebooks/make_dataset_scripts/process_data.ipynb`):
   - Паркетные паспорта превращаются в `*.pkl.gz` словари, где ключ — индекс, а значение — словарь категориальных и числовых признаков. Там же сохраняются тестовые словари (`test_dict.pkl.gz`) и веса для негативного сэмплинга.
3. **Стриминг датасета** (`src/datasets/stream_dataset.py`):
   - Датасет итерируется по чанкам Parquet (через `pyarrow.ParquetFile.iter_batches`). Для каждого положительного `(inn_dt, inn_kt)` в батч генерируется пара с `label=1` и случайный негатив с `label=0`. Фичи берутся из подготовленных словарей, недостающие значения заполняются нулями.
   - Коллатер (`src/datasets/collate.py`) собирает батч, который содержит категориальные/вещ. признаки и предрасчитанные 256-мерные эмбеддинги.

## Архитектура модели
- **Категориальные фичи** пользователей и айтемов проходят через `FeatureEmbedding` с отдельными слоями на каждый признак и ограничением размерности `min(sqrt(cardinality), max_embed_dim)`.
- **Проекция фичей**: слои `FeatureProjection` выравнивают все эмбеддинги в общее пространство `embed_dim` (по умолчанию 128).
- **Численные признаки** (`double_user`, `double_item`) обрабатываются батч-нормализацией + линейными слоями.
- **Внешние эмбеддинги** dt/kt проецируются в то же пространство (`dt_emb_proj` и `kt_emb_proj`).
- **Внимание внутри башен**: пользовательская и товарная башни используют `AttentionLayer`, который учится взвешивать вклад каждого признака и эмбеддинга. Результирующие векторы нормализуются и скалярно перемножаются, образуя `logits` для BCE-лосса (`src/loss/bceloss.py`).
- **Трейнер** (`src/trainer/trainer.py`) наследует `BaseTrainer`, логирует метрики (accuracy, ROC-AUC, PR-AUC, F1 валидации), клипает градиенты и пишет attention-карты в TensorBoard.

## Обучение
1. Установите зависимости:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Подготовьте конфиги: укажите пути к parquet и словарям в `src/configs/datasets/*.yaml`, а также `data.user_feature_sizes_path` и `data.item_feature_sizes_path` в `src/configs/test.yaml`.
3. Запустите тренировку (пример для набора `transactions_21`):
   ```bash
   python3 train.py \
     trainer.n_epochs=5 \
     trainer.log_step=1000 \
     data.user_feature_sizes_path=/path/to/user_feature_sizes.json \
     data.item_feature_sizes_path=/path/to/item_feature_sizes.json \
     datasets=transactions_21 \
     model.embed_dim=128 \
     optimizer.lr=3e-4
   ```
   Hydra положит логи и чекпоинты в `deepfm_logs/<run_name>` и (опционально) в W&B/TensorBoard.
4. Для инференса и пересчёта метрик подготовьте отдельный конфиг `src/configs/inference.yaml` и вызовите `python3 inference.py inferencer.from_pretrained=<ckpt>`.

## Постобработка, эмбеддинги и выборка кандидатов
1. **Съём эмбеддингов** (`notebooks/process_results/collect_embeddings.ipynb`):
   - Загружает обученную модель и словари фичей, прогоняет все dt/kt из тестового словаря и сохраняет 128-мерные эмбеддинги и attention-веса в `../data/<version>/embeddings/*.pkl.gz`.
2. **Построение ANN-индекса** (`notebooks/make_recommendations/recommendations_from_embeddings.ipynb`):
   - Сканирует словари эмбеддингов, строит Annoy-индекс (`top_k`, `n_trees` задаются в ноутбуке) и сохраняет рекомендации (`recommendations/*.pkl.gz`). Сейчас это делается только для тестового среза.
3. **Офлайн-метрики** (`notebooks/make_recommendations/recommenders_metrics.ipynb`):
   - Берёт `test_dict.pkl.gz`, набор рекомендаций и считает `MAP@k`, `Precision@k`, `Recall@k`, `NDCG@k`. Есть визуализация метрик по k. Это основной инструмент sanity-check перед выкладкой.

## Рекомендации по запуску end-to-end
1. Выполнить `notebooks/make_dataset_scripts/run.ipynb`, передав корректный `config.yaml`, чтобы собрать train/test паркет и словари.
2. Выполнить `notebooks/make_dataset_scripts/process_data.ipynb`, чтобы упаковать признаки и индексы в `*.pkl.gz`.
3. Настроить `src/configs/datasets/<dataset>.yaml` так, чтобы оно указывало на свежесгенерированные файлы.
4. Запустить `train.py` с нужными override-параметрами.
5. Через `collect_embeddings.ipynb` получить эмбеддинги dt/kt для интересующего множества пользователей (пока только hold-out).
6. Через `recommendations_from_embeddings.ipynb` построить Annoy/FAISS индекс и выгрузить top-k кандидатов.
7. Через `recommenders_metrics.ipynb` посчитать метрики и сравнить с бенчмарком.

## Направления для улучшения и автоматизации
1. **Убрать зависимость от ноутбуков**:
   - `notebooks/make_dataset_scripts/run.ipynb` и `process_data.ipynb` содержат чистый PySpark/Pandas код, а конфиг уже вынесен в YAML. Их можно перенести в модуль `src/pipelines/datasets/{prepare,pack}.py` с CLI (`python3 -m src.pipelines.datasets.prepare +config=config.yaml`), чтобы запускалось как регулярная Spark-джоба и попадало в планировщик (Airflow, Oozie).
   - `notebooks/process_results/collect_embeddings.ipynb`, `notebooks/make_recommendations/recommendations_from_embeddings.ipynb` и `recommenders_metrics.ipynb` так же стоит оформить как скрипты (`src/pipelines/export_embeddings.py`, `src/pipelines/build_index.py`, `src/pipelines/evaluate.py`). Это позволит автоматически пересчитывать эмбеддинги для **всех** пользователей перед выкладкой, а не только для тестовых dt.
2. **Пайплайн «данные → модель → рекомендации»**:
   - Завести единый Hydra/MLflow workflow либо makefile, который последовательно выполняет подготовку данных, обучение и инференс, чтобы исключить ручные шаги.
3. **Негативный сэмплинг**:
   - В `StreamDataset` (`src/datasets/stream_dataset.py`) негативы берутся через `random.choice` по `unique_kt`. В `process_data.ipynb` уже готовятся `sampling_arrays.pkl.gz` с весами, но они не используются. Стоит добавить возможность подгружать веса и сэмплировать пропорционально частоте/унитарности, чтобы уменьшить шум и переобучение на популярных kt.
4. **Масштабирование инференса**:
   - Сейчас `collect_embeddings.ipynb` прогоняет пользователей в цикле по одному и держит словари в памяти. Можно вынести код в модуль, организовать батчи через `DataLoader` и разделить прогон на несколько GPU/CPU-воркеров. Это также упростит получение эмбеддингов для всех клиентов при выкладке модели.
5. **Автогенерация ANN индекса**:
   - Строительство Annoy-индекса сейчас происходит вручную в ноутбуке. Скрипт с аргументами (`input_embeddings`, `top_k`, `backend`) позволит запускать построение индекса из пайплайна и автоматически сохранять версию в хранилище.
6. **Тесты и мониторинг фичей**:
   - Добавить smoke-тесты для `StreamDataset` (некоторые данные можно синтетически сгенерировать) и валидацию схем (например, через `pydantic` или `pandera`). Это поможет ловить несоответствия между Spark-выгрузками и ожиданиями модели.
7. **Конфигурация и документация**:
   - Сейчас `src/configs/test.yaml` использует статический набор путей. Имеет смысл описать несколько типовых пресетов (`prod.yaml`, `offline.yaml`) и продокументировать необходимые поля конфигурации (размерности эмбеддингов, список признаков), чтобы при переносе на новую временную выборку не приходилось искать нужные места вручную.

Эти шаги позволят воспроизводимо готовить данные, тренировать модель и выпускать эмбеддинги для всех пользователей без ручных действий в ноутбуках, что критично для будущей автоматизации выкладки кандидатогенератора.
