# sberindex-clustering

Решение задачи «Кластеризация» онлайн-конкурса СберИндекс 2026.

Строим динамическую атрибутированную сеть муниципальных образований (МО) России. Узлы сети это МО, атрибуты узлов это экономические характеристики, рёбра это экономическая близость. Сеть строится по кварталам 2023 и 2024 годов. Затем МО кластеризуются, изменения кластеров отслеживаются во времени, а кластеры описываются как типы местных экономик.

## Установка

```
python -m venv .venv
.venv/Scripts/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
```

PyTorch нужен только для метода DMoN, достаточно версии для процессора. В Linux и macOS путь к pip `.venv/bin/pip`, без отдельной первой команды pip в Linux поставит PyTorch с CUDA (несколько гигабайт).

## Запуск

```
python run.py download
python run.py features
python run.py compare
python run.py track
python run.py robustness
python run.py describe
python run.py validate
```

| Шаг | Что делает | Результат |
|---|---|---|
| `download` | скачивает пакет данных конкурса (20 МБ) и индекс мобильности | `data/raw` |
| `features` | собирает признаки МО по кварталам и по скользящему году | `data/processed` |
| `compare` | сравнивает правила рёбер и методы кластеризации, около 75 минут | `results` |
| `track` | строит типы МО по окнам и прослеживает их во времени, около 2 минут | `results` |
| `robustness` | проверяет устойчивость типов к параметрам и составу МО, около 50 минут | `results` |
| `describe` | паспорта типов, изменения, внешняя проверка, карты и графики, около 20 секунд | `results`, `results/figures` |
| `validate` | проверяет типы: признаки, сеть, мосты между типами, согласие методов, устойчивость, ранжирование методов, около 20 минут | `results`, `results/figures` |

Таблицы Росстата, цены по регионам и справочник МО уже лежат в `data/prepared`. Чтобы пересобрать их из первоисточников (ещё около 5 ГБ загрузок), есть `python run.py download --all` и `python run.py prepare`, подробности в [docs/data.md](docs/data.md).

## Документация

- [docs/data.md](docs/data.md) источники данных, лицензии, подготовленные таблицы
- [docs/features.md](docs/features.md) признаки МО
- [docs/network.md](docs/network.md) построение сети и правило рёбер
- [docs/methods.md](docs/methods.md) методы кластеризации
- [docs/metrics.md](docs/metrics.md) метрики качества кластеров
- [docs/dynamics.md](docs/dynamics.md) отслеживание типов во времени
- [docs/robustness.md](docs/robustness.md) устойчивость результатов
- [docs/validation.md](docs/validation.md) проверка результатов

## Структура

- `run.py` точка входа
- `config.yaml` источники данных, показатели Росстата, сопоставление названий регионов, параметры сети, модели, сравнения, динамики, проверок и описания типов
- `src/download.py` загрузка исходных данных
- `src/prepare.py` подготовка таблиц Росстата, цен и справочника МО
- `src/features.py` таблица признаков
- `src/network.py` отбор узлов, преобразование признаков, правила рёбер
- `src/kefrin.py` метод KEFRiN
- `src/canus.py` метод CANUS
- `src/dmon.py` метод DMoN
- `src/methods.py` методы кластеризации
- `src/metrics.py` метрики качества кластеров
- `src/compare.py` сравнение правил рёбер и методов
- `src/dynamics.py` отслеживание типов во времени
- `src/robustness.py` проверки устойчивости
- `src/describe.py` паспорта типов и графики
- `src/validate.py` проверка результатов
- `src/api.py` загрузка данных через API СберИндекса
- `data/prepared` подготовленные данные
- `docs` документация

## Лицензия

Код распространяется по лицензии MIT, см. [LICENSE](LICENSE).

Данные в `data` распространяются на условиях их источников: данные СберИндекса и таблицы, сделанные из них, по CC BY-SA 4.0, данные Росстата в обработке проекта «Точно» по CC BY 4.0. Подробности и ссылки для цитирования в [docs/data.md](docs/data.md).