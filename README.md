# DeepFM модель рекомендаций

При возникновении вопросов пишите в tg @aapetukhov

Этот репозиторий содержит код и артефакты (не в BitBucket, но в hdfs) подготовки и обработки данных, обучения, инференса и подсчёта метрик рекомендательной модели. Модель по сути представляет из себя кандидатогенератор и стремится покрывать наибольшее количество реальных взаимодействий в первых 100 рекомендациях. Модель `DeepFM` (архитектура лежит в `src/models/deepfm.py`) учится на транзакциях и строит эмбеддинги для плательщиков (`dt`) и получателей (`kt`). В прямом направлении модели естественным образом обозначаются `dt` как user, а `kt` - как item, поскольку `dt` платит деньги (как обычно делает пользователь сервиса), а `kt` как item их получает. Данные собираются в Spark-среде (`notebooks/make_dataset_scripts/run.ipynb`), упаковываются в компактные словари (`process_data.ipynb`), после чего попадают в стриминговый датасет (`src/datasets/stream_dataset.py`) и цикл обучения. Ключевое место, где стартует обучения - это `train.py`. Обучение hydra-powered, чтобы параметры можно было передавать прямо через командную строку без прибегания к `argparse`. Офлайн-инференс и метрики считаются централизовано через `src/pipelines/offline_inference.py` / `inference.py`.

## Как познакомиться с репозиторием
Кодовая база весьма разностороння, но в сути своей содержит скрипты сборки и обработки датасета, обучения модели и её инференса, по большей части в Python-скриптах (хотя датасеты собираются в IPython). Поэтому лучше всего будет внимательно читать код. Всё же, рекомендуемый порядок мне видится таким:
1. **Пройтись по Spark-пайплайну.** Изучите `notebooks/make_dataset_scripts/run.ipynb`, `config.yaml` и `utils.py`, чтобы понять, откуда приезжает каждая таблица и какие артефакты кладутся в HDFS. Артефакты с этого этапа высыпаются в HUE в персональное хранилище в `/user/YOUR_NUMBER_omega-sbrf-ru` и включают в себя таблицы взаимодействий, наборы фичей, индексаторы для подачи в эмбеддинг-слой модели и проч. Полный список и более подробное описание см. ниже.
2. **Разобраться с упаковкой признаков.** Данные из HUE переносятся в Jupyterhub (Datalab среда с GPU по адресу `jupyterhub-datalab.apps.prom-datalab.ca.sbrf.ru`) командой `hdfs get` (подробнее ниже) и обрабатываются в подходящий для обучения формат в `notebooks/process_data.ipynb`. Его нужно прогонять осторожно и если ядро сдыхает или замерзает, то перезапускать и запускать с того момента, где остановились. В этом ноутбуке parquet файлы обрабатываются в словари и `*_feature_sizes.json`, а также образуются тестовые словари для замера метрик и негатив семплинга во время обучения (`test_dict.pkl.gz` и `unique_inn_*.pkl.gz`). Вручную нужно будет редактировать конфиг датасета `src/configs/datasets/*.yaml`.
3. **Посмотреть подачу данных в датасет.** `src/datasets/stream_dataset.py` описывает потоковый `IterableDataset`, который читает parquet чанками, матчит `inn_*` с индексами, добавляет негативы и собирает батчи. `src/datasets/collate.py` собирает всё в словарь и батч плывёт в таком виде дальше по циклу обучения. Словарь, хранящий батч, по пути обогащается через update (см, например, `/src/trainer/trainer.py`).
4. **Изучить код модели и обучения** Исходный код модели и её архитектура лежит в `src/models/deepfm.py` и представляет из себя двухбашенную модель, которая максимизирует близость эмбеддингов через выбираемый лосс. Оптимальным по скорости обучения и качеству результатов принят BCE, который лежит в `src/loss/bceloss.py`. Обучение происходит в `src/trainer/trainer.py` и `src/trainer/base_trainer.py` - там реализуется стандартный training loop, с разбивкой по батчам и безопасным сохранением лучшей версии модели на случай, если обучение прерывается через KeyBoardInterrupt или из-за нехватки памяти CUDA. Обучение логируется в консоль и в файл с логами, артефакты потом высыпаются в директорию `deepfm_logs/YOUR_RUN_NAME/` и содержат `model_best.pth`, полный конфиг и лог. Обучение логируется в лайве в `tensorboard` на localhost, при запуске обучения в консоли вылезает зелёного цвета ссылка, где можно мониторить обучение в прямом эфире.
5. **Изучить код офлайн-инференса.** Единый скрипт обучения, который выгружает эмбеддинги, строит Annoy-индекс и считает метрики, лежит в `src/pipelines/offline_inference.py`. Он вызывается через `python inference.py --config-name="inference_24.yaml"`, например. Инференс, как и об
6. **Изучить конфиги Hydra.** Директория `src/configs/` хранит пресеты датасетов, моделей, трейнера и инференса. При обучении я передаю в `train.py` / `inference.py` как правило только название очередного конфига, поэтому ознакомьтесь обязательно с `src/configs/test_*.yaml` и `inference_*.yaml` как с примерами рабочих запусков.

## Структура репозитория
- `src/` - код (датасеты, модели, лоссы, трейнер, офлайн-пайплайны, логгеры и Hydra-конфиги).
- `notebooks/make_dataset_scripts/` - PySpark-пайплайны подготовки данных (`run.ipynb`, `process_data.ipynb`, `utils.py`, `config.yaml`).
- `train.py`, `inference.py` - основные CLI-входные точки Hydra.
- `data/`, `deepfm_logs/`, `tensorboard/` - сюда складываются данные и артефакты (паркетники, словари, чекпоинты, логи, значения метрик).

## Подготовка данных (детальный воркфлоу)
### 1. Spark-ноутбук `run.ipynb`
Ноутбук работает по ячейкам, каждая ячейка собирает свою значимую часть датасета и вырубает спарк-контекст, следующая ячейка его стартует заново с теми же параметрами, которые указываются в конфиге. Сделано это для того, чтобы спарк-контекст не умирал каждый раз.
- **Конфиг:** `notebooks/make_dataset_scripts/config.yaml` задаёт Spark-параметры, окна train/test, пути до внешних эмбеддингов (`paths.embeddings`) и контекстных слов из транзакций (`paths.context_words`), а также схему сохранения `outputs.save_schema`. В финальной версии модели из источников используются:
    - `arnsdpsbx_t_team_apm.tmb_basis_client` (см `utils.py`),
    - `basis_transactions_coloured` (см `utils.py`)
    - Эмбеддинги Постновой (путь сложночитаемый, тк это подписка в СМД) `hdfs://hdfsgw/arnsdpcc360__Podpiska_na_produkty_Postnovoj-CUSTOM_CIB_ML360-MON_AI_UL_EMBEDDING_V2/data/custom/cib/ml360/pa/mon_ai_ul_embedding_v2/mon=25-08-31` (см `config.yaml`)
    - Извлеченные из транзакций слова `arnsdpcc360__smd_recsys_products-CUSTOM_CIB_ML360_CLIENTS_PRODUCTS_EXTRACT-TRANSACTIONS_PRODUCTS_EXTRACT/data/custom/cib/ml360_clients_products_extract/pa/transactions_products_extract`
- **Part 1:** В части 1 выкачиваются транзакции за заданные даты, перемешивается train и сохраняется `save_schema.train_interactions` / `.test_interactions` в личное пространство в HUE (hive).
- **Part 2:** Считаются числовые агрегаты по суммам платежей для dt/kt, формируются таблицы `*.dt_stats`, `*.kt_stats`.
- **Part 3:** Из ОКВЭД/ОКАТО/БИК формируются декомпозированные признаки (`*_lvl1...lvl4`, `bic_*_34/56/79`), объединяются с числовыми агрегатами.
- **Part 4:** Все категориальные признаки (включая сами значения `inn_dt`/`inn_kt`) индексируются `StringIndexer`-моделями, результат сохраняется в `*.dt_pass_indexed`, `*.kt_pass_indexed`, а модели - в `*.indexer_dt`, `*.indexer_kt`.
- **Part 5:** Изначально была и использовалась для негативного сэмплинга и hard negatives (используются в `process_data.ipynb` / датасете), но была упразднена из-за того, что так модель училась слиишком медленно и не давало буста в качестве.
- **Part 6:** Графовые эмбеддинги Постновой подтягиваются из HDFS (`paths.embeddings`) и присоединяются к индексированным паспортам (`*.dt_embeddings_indexed`, `*.kt_embeddings_indexed`).
- **Part 7:** Собираются тестовые словари (`*.test_dict`, `*.test_reverse_dict`), фильтруются по `filters.min_test_cnt_*` - это поле из конфига для минимального числа транзакций, совершенных пользователем, чтобы быть включенным в тестовую выборку.
- **Topic LDA:** Тематическое моделирование выполнено с помоью LDA-модели на очищенных транзакционных сообщениях. Функция `build_topic_embeddings` строит `context_{dt,kt}_cv_model`, `context_{dt,kt}_lda_model` и паркетник с topic-весами (`*_topic_embeddings`), получая таким образом эмбеддинг каждого пользователя на основе тех сообщений, которые он отправлял.

На выходе `run.ipynb` в HDFS/`outputs.data_dir` лежат все parquet-артефакты, которые потом понадобятся PyTorch-части. Последние ячейки ноутбука переносят эти файлы в ту область, из которой их можно будет перенести в Jupyterhub. Пояснение: файлы сперва кладутся в `viewfs://...`, откуда их невозможно командой перетащить в Jupyterhub.

Перед тем, как прогонять эти ячейки, данные и артефакты необходимо вручную переместить в директорию `user/YOUR_NUMBER_omega-sbrf-rudeepfm_data/data_24` (24 взято для примера как последняя итерация моей модели, подставьте своё значение). Далее выполните последние ячейки, в которых содержится команда
```
# AFTER MOVING FILES ABOVE TO A NEW DIR
!hdfs dfs - -mkdir hdfs://arnsdpsbx/user/YOUR_NUMBER_omega-sbrf-ru/data/data_24
!hdfs dfs -cp viewfs://SDP-leverkin-ap-ca-sbrf-ru-SberSovetnik-471b9a/user/YOUR_NUMBER_omega-sbrf-ru/deepfm_data/data_24/* hdfs://arnsdpsbx/user/YOUR_NUMBER_omega-sbrf-ru/data/data_24
```

После выполнения этой команды перейдите в Jupyterhub и в терминале выполните команду, которая перенесёт эти файлы в локальную директорию (файлы тяжёлые, так что команда будет выполняться несколько минут):
```
hdfs dfs -get hdfs://arnsdpsbx/.../data/data_24/ deepfm/data/train_24/
```


P.S. Из скриптов видно, что данные перетекают в проиндексированном виде, а не в сыром (вместо конкретного значения inn_dt в модель подаётся индекс, например 346732). Это связано с тем, что эмбеддинг-слой в PyTorch умеет обрабатывать только проиндексированные значения категориальных фичей, а не сырые строки, поэтому важно сохранить индексаторы для использования в проме.

### 2. `process_data.ipynb`
- Глобальная переменная `version` задаёт, какую директорию из `data/<version>` упаковывать, в нашем случае это 24. Необходимо ВРУЧНУЮ поменять везде в ноутбуке на 24, поскольку где-то ноутбук подыхал и приходилось вручную прописывать пути. При запуске от начала до конца всё должно работать исправно, но ноутбук вряд ли выдерит из-за особенностей дефолтных настроек Jupyter в этой среде. Как вариант, это можно перенести в отдельный `process_data.py` скрипт.
- Для dt/kt читаются `*.pass_indexed` parquet, формируется словарь `{index: feature_dict}` и сохраняется в `dt_features_dict.pkl.gz` / `kt_features_dict.pkl.gz`.
- Одновременно собираются `user_feature_sizes.json` / `item_feature_sizes.json` (кол-во уникальных значений каждого категориального признака, нужное для инициализации эмбеддинг-слоя модели), и через `validate_feature_sizes` проверяется соответствие `src/configs/datasets/<dataset>.yaml`.
- `test_dict.pkl.gz` и `test_reverse_dict.pkl.gz` строятся из parquet `*.test_dict` / `*.test_reverse_dict`.
- `dt_embeddings_dict.pkl.gz`, `kt_embeddings_dict.pkl.gz`, `dt_topic_embeddings_dict.pkl.gz`, `kt_topic_embeddings_dict.pkl.gz` и `unique_inn_{dt,kt}.pkl.gz` собираются по соответствующим parquet/индексерам - эти словари сразу используются `StreamDataset`.
- В итоге получаем в директории `data/<version>/` (в нашем случае `data/train_24`) полный набор артефактов, необходимых для обучения. Эти `*.pkl.gz`, `feature_sizes.json` и parquet-папки, которые прописываются в Hydra-конфигах (`src/configs/datasets/train_24.yaml` и т.п.).

### 3. Датасет (`src/datasets/stream_dataset.py`)
- В конструктор передаются пути из dataset-конфига: директория parquet (`parquet_dir`), словари признаков/эмбеддингов, маппинги `inn_dt/kt -> dt/kt_index` и списки уникальных `dt/kt`.
- В функции `__iter__` parquet читаются чанками через `pyarrow.ParquetFile.iter_batches`, каждая строка превращается в позитивный пример.
- Для каждого позитива генерируется негатив - случайный `inn_kt` (или `inn_dt` в reverse версии) с label=0.
- Функция `make_sample` собирает словарь с категориальными и числовыми признаки, базовыми эмбеддингами (`dt_emb`, `kt_emb`), topic-эмбеддингами и label. Недостающие значения заполняются нулями. collate-функция в `src/datasets/collate.py` превращает их в батч.

## Архитектура модели (`src/models/deepfm.py`)
В архитектурной составляющей модели есть две идентичные башни, обрабатывающие одна `kt`, а другая - `dt`, идентичным образом: все фичи преобразовываются в последовательность векторов и обрабатываются Attention-слоем, за которым следует взвешенная агрегация, превращающая последовательность в вектор. Два вектора (по одному от `kt` и `dt`) скалярно перемножаются, выдавая их similarity score.
- **Embedding + Projection.** `FeatureEmbedding` строит отдельный `nn.Embedding` на каждый категориальный признак. Поскольку фичи имеют разную кардинальность и сделать один общий размер эмбеддинга под все фичи было бы неразумно, то размерность эмбеддинг-слоя конкретной фичи выбирается как `min(sqrt(cardinality), max_embed_dim)`. Линейный слой `FeatureProjection` проецирует их в общее пространство `embed_dim`.
- **Непрерывные признаки.** Непрерывные фичи союираются в вектор и переводятся в общую размерность `embed_dim` через `user_double_proj` / `item_double_proj` - последовательность слоёв BatchNorm, Linear, BatchNorm,  ReLU.
- **Эмбеддинги Постновой и topic-эмбеддинги.** Слои `dt_emb_proj`, `kt_emb_proj`, `dt_topic_emb_proj`, `kt_topic_emb_proj` приводят 256-мерные и `n_topics`-мерные вектора к общей размерности `embed_dim`. То есть эмбеддинги постновой
- **AttentionLayer.** Для каждой башни вычисляется self-attention, затем дополнительный attention (`W_u`) выделяет важность фичей и эмбеддингов для конкретного пользователя - это моя интерпретация Attention Pooling. Выход нормализуется через `LayerNorm`.
- **Скоринг.** Пользовательский и товарный вектор перемножаются скалярно и отдаются в `BCEWithLogitsLoss` (`src/loss/bceloss.py`).
- **Трейнер.** Далее в процессе обучения трейнер (`src/trainer/trainer.py`) логирует accuracy/ROC-AUC/PR-AUC/F1, поддерживает градиентный клиппинг, TensorBoard/Comet/W&B и early stopping.

## Обучение
Если уже имеется готовая обучающая выборка и вы хотите, например, обучить модель на бОльшем количестве эпох, то вам предлагается сделать следующие шаги:
1. **Настроить окружение.** Для настройки окружения рекомендую конфигурировать `pip` (я специально положил пример своего конфига, чтобы было проще разобраться) и активировать `.bashrc` файл, чтобы установить необходимые зависимости - небходимо посмотреть их солержимое. Файл .bashrc стоит заранее положить в nfs/ и конфигурировать `pip` для возможности установки библиотек из сберовского зеркала PyPi (поскольку в омеге доступа к PyPi, очевидно, нет). В `.bashrc` файле также содержатся удобные комнды для просмотра размеров файликов в директории в МБ и ГБ, а также другие полезные в быту команды.
   ```bash
   # эту команду лучше выполнять из nfs/
   cd nfs
   source .bashrc
   # а эту команду - из nfs/deepfm/
   cd deepfm
   pip install -r requirements.txt
   ```
2. **Конфигурация.** Скопируйте актуальный dataset-пресет (`src/configs/datasets/train_24.yaml` и т.п.) и убедитесь, что:
   - `train.parquet_dir`, `train.dt_feat_path`, `train.kt_feat_path`, `train.dt_emb_path`, `train.kt_emb_path`, `train.dt_topic_emb_path`, `train.kt_topic_emb_path`, `train.inn_dt_to_idx_path`, `train.inn_kt_to_idx_path`, `train.unique_dt_path`, `train.unique_kt_path` ведут к свежим файлам из `data/<version>`.
   - `data.user_feature_sizes_path`, `data.item_feature_sizes_path`, `data.test_dict_path` прописаны в `src/configs/test_*.yaml`.
   - Измените желаемое число эпох или что вам вздумалось в `src/configs/test_*.yaml`.
3. **Запуск.**
   ```bash
   python train.py \
     --config-name test_24 \
     datasets=train_24 \ #это опционально, можно не писать
     trainer.n_epochs=5 \ #можно указать число эпох прямо в командной строке
     trainer.log_step=1000 \ #частота логирования
     data.user_feature_sizes_path=data/train_24/user_feature_sizes.json \
     data.item_feature_sizes_path=data/train_24/item_feature_sizes.json \
     optimizer.lr=3e-4 \ #параметры оптимайзера
     model.embed_dim=128 #выходной размер эмбеддинга пользователя
   ```

   Или достаточно простого
   ```bash
   python train.py --config-name="...test_24"
   ```

   Чекпоинты и Hydra-конфиги окажутся в `deepfm_logs/<run_name>`, TensorBoard - в `tensorboard/<dataset>/<run>`.
   Одна эпоха на двухмесячном датасете занимает порядка 4-5 часов обучения + столько же на валидацию после эпохи. Имейте в виду, что после примерно 24 часов бездействия JupyterHub автоматически вырубается, даже если там крутится обучение, поэтому нужно регулярно заходить, чтобы убедиться, не сдохло ли обучение. К сожалению, виртуалки у нас работают только так :) Можно поставить `save_period` на 1, чтобы сохранялся результат каждой эпохи, но модели довольно много весят и есть риск забарахлить директорию.

   Модели, получившиеся у меня, содержат порядка 500М (500 миллионов) параметров и весят в районе 6 ГБ. Две модели (от kt к dt и от dt к kt) суммарно, таким образом, весят 12 ГБ.

## Инференс и офлайн-метрики
После обучения можно сразу же запускать инференс
- Команда `python inference.py --config-name="..."` запускает `src/pipelines/offline_inference.py`, который выполняет: экспорт эмбеддингов, построение Annoy-индекса (приближенный индекс ближайших соседей для быстрого поиска в продакшне), подсчёт `MAP@k`, `Precision@k`, `Recall@k`, `NDCG@k` и сохранение `metrics.json`.
- Базовый конфиг: `src/configs/inference.yaml` (или `inference_24.yaml`, `inference_reverse_24.yaml`). Через CLI можно переопределить `datasets=...`, `model=...`, `inference.checkpoint_path=...`, `inference.io.output_dir=...`, `inference.user_ids_source` (`all`, `test_dict`, путь до `.pkl.gz`), `inference.item_ids_source`, параметры Annoy (`inference.ann.*`) и `metrics_k`.
- Пример запуска (прямое направление):
  ```bash
  python inference.py \
    --config-name inference_24 \
    inference.checkpoint_path=deepfm_logs/train_24_run_1/model_best.pth \ # путь к обученной модели
    inference.io.output_dir=data/train_24/offline_inference_direct \ # путь для выходящих артефактов
    inference.user_ids_source=test_dict \ #откуда берутся айдишники пользователей
    inference.save_attentions=true \ #сохранять ли важности фичей для пользователей при расчёте
    inference.top_k=100 \
    inference.metrics_k=[20,50,100] #для каких значений k считать metric@k
  ```
  Можно ограничиться простым
  ```bash
  python inference.py --config-name="inference_24.yaml"
  ```

- Пример для обратного направления `от kt к dt`:
  ```bash
  python inference.py \
    --config-name inference_reverse_24 \
    model=deepfm_reverse \
    datasets=train_24_reverse \
    data.user_feature_sizes_path=data/train_24/item_feature_sizes.json \
    data.item_feature_sizes_path=data/train_24/user_feature_sizes.json \
    inference.user_entity=kt \
    inference.item_entity=dt
  ```
- В конфиге в `inference.io.*` задаются файлы (`user_embeddings`, `item_embeddings`, `recommendations`, `metrics`). При `inference.reuse_artifacts=true` скрипт не пересчитывает уже существующие артефакты.
- Код офлайн-пайплайна можно вызвать напрямую через `python -m src.pipelines.offline_inference +inference=<overrides>` - это упрощает интеграцию в Airflow/Oozie.

## Что меняем вручную и что подтягивается автоматически
- **Вручную:**
  - `notebooks/make_dataset_scripts/config.yaml` - окна дат, пути в HDFS, схема сохранения (при необходимости), Spark-ресурсы для сессии.
  - `report_dt` в функции `filter_inn` в `utils.py` - при устаревании. Этот скрипт ещё от Никиты.
  - После сборки датасета в run.ipynb данные и артефакты необходимо вручную переместить в директорию `user/YOUR_NUMBER_omega-sbrf-ru/deepfm_data/data_24` (24 взято для примера как последняя итерация моей модели, подставьте своё значение). Далее выполните последние ячейки ноутбука, в которых и написано это предупреждение в виде комментария перед bash-командой.
  - `process_data.ipynb` - переменная `version` + пути к parquet-файлам.
  - Hydra-конфиги (`src/configs/datasets/*.yaml`, `src/configs/test_*.yaml`, `src/configs/inference*.yaml`) - прописываем пути к новому набору файлов и пресеты модели/трейнера.
  - CLI overrides при запуске `train.py`/`inference.py` (скорость обучения, эпохи, пути до чекпоинтов).
- **Автоматически:**
  - Внутри `run.ipynb` пути строятся от `outputs.save_schema`; достаточно сменить `save_schema`, чтобы перезаписать весь набор артефактов с новым префиксом.
  - `process_data.ipynb` использует стандартные имена (`arnsdpsbx_t_team_fin_adviser.*`) и складывает `*.pkl.gz`/JSON в `data/<version>` без ручного перечисления файлов - достаточно обновить `version`.
  - `StreamDataset` сам подхватывает списки уникальных dt/kt и словари признаков по путям, указанным в dataset-конфиге.
  - `inference.py` умеет переиспользовать существующие эмбеддинги/рекомендации (`inference.reuse_artifacts`), поэтому при повторных запусках не нужно менять код.

## Рекомендации по запуску end-to-end
1. **Spark (run.ipynb).** Сперва настроить `config.yaml`, прогнать ноутбук, дождаться parquet и логов в `outputs.data_dir`. Перенести всё это в Jupyterhub
2. **Упаковка (process_data.ipynb).** Сгенерировать артефакты, необходимые для обучения - `*.pkl.gz`, `feature_sizes.json`, `unique_inn_*.pkl.gz` с помощью этого ноутбука.
3. **Конфиги / проверка.** Обновить `src/configs/datasets/<dataset>.yaml`, `src/configs/test_*.yaml`, и убедиться, что `validate_feature_sizes` проходит.
4. **Обучение (`train.py`).** Запустить обучение, смотреть логи в консоли, в `deepfm_logs/<run>` и в TensorBoard.
5. **Инференс (`inference.py`).** Выгрузить эмбеддинги, Annoy и метрики, проверить `metrics.json`.
6. **Передача дальше.** `data/<version>/offline_inference_*` содержит `embeddings/*.pkl.gz`, `recommendations/*.pkl.gz`, `metrics.json` - эти артефакты по идее должны будут в будущем передаваться следующей системе и крутиться в онлайн-сервисе (FAISS/Redis/онлайн-сервис).

## Обучение в обратном направлении (reverse, когда для kt мы рекомендуем dt)
Модель симметрична: `StreamDatasetReverse`, `DeepFMReverse` и `bceloss_reverse` повторяют прямой pipeline, но меняют роли башен. Потребуется:
1. (Опционально) запуск `run.ipynb`/`process_data.ipynb` с `config_reverse.yaml`, чтобы получить отдельные parquet/словарные файлы.
2. `python train.py --config-name test_reverse ...` с переставленными `*_feature_sizes`.
3. `python inference.py model=deepfm_reverse datasets=... inference.user_entity=kt inference.item_entity=dt`.
Все остальные шаги идентичны; данные можно переиспользовать, если `inn_dt_index`/`inn_kt_index` уже рассчитаны.

## P.S. Направления для улучшения и автоматизации
1. **Отказ от ноутбуков.** `run.ipynb` и `process_data.ipynb` можно вынести в `src/pipelines/datasets/{prepare,pack}.py`, чтобы запускать Spark-джобы напрямую.
2. **Автоматизированный inference loop.** Написать inference loop на основе существующего `offline_inference.py` - из этого кода понятно, как следует инициализировать модель и подавать в неё данные для инференса.
3. **Масштабирование инференса.** Выделить батчевый экспорт эмбеддингов, чтобы параллелить работу по CPU/GPU и не держать все словари в памяти ноутбука.
4. **Авто-ANN.** Неплохо было бы версионировать индексы, поэтому хорошо было бы заиметь автоматический скрипт для построения Annoy/FAISS по входным эмбеддингам, который избавит от ручного запуска ноутбуков.
