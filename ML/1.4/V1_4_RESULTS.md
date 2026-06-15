# ML v1.4 - результаты LightGBM уточняющая проверка

Эксперимент `v1.4` проверял, можно ли закрепить и улучшить результат `v1.3` без нового feature engineering. Датасет и набор признаков остались теми же: `client_profile_v1_0_shuffled.csv`, `full_base`, исключение `client_id`, прямых fraud-утечек и признаков с долей пропусков выше 95%. Поэтому выводы `v1.4` относятся именно к режиму обучения LightGBM и выбору метрик, а не к новой витрине данных.

Главный результат: лучшей моделью стала `lgbm_auc_unweighted_seed2026`. Она оказалась лучшей одновременно по `val PR-AUC`, `val ROC-AUC`, `test PR-AUC` и `test ROC-AUC`. На test она дала `PR-AUC = 0.4967` и `ROC-AUC = 0.8987`. Это заметное улучшение относительно `v1.3 lightgbm_default`, где test `PR-AUC = 0.4779` и `ROC-AUC = 0.8687`. Также это выше лучшего исторического CatBoost long-run из `v1.1`, где test `PR-AUC = 0.4708` и `ROC-AUC = 0.8903`.

## Общая таблица лучших test-результатов

| Model | Test ROC-AUC | Test PR-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 | Positive rate |
|---|---:|---:|---:|---:|---:|---:|
| `lgbm_auc_unweighted_seed2026` | 0.8987 | 0.4967 | 0.7529 | 0.2659 | 0.3930 | 0.0132 |
| `lgbm_auc_unweighted_seed42` | 0.8985 | 0.4940 | 0.7371 | 0.2649 | 0.3897 | 0.0134 |
| `lgbm_auc_unweighted_seed777` | 0.8984 | 0.4919 | 0.7293 | 0.2628 | 0.3864 | 0.0134 |
| `lgbm_pr_default_seed777` | 0.8665 | 0.4809 | 0.3217 | 0.6129 | 0.4219 | 0.0710 |
| `lgbm_pr_default_seed42` | 0.8703 | 0.4802 | 0.3130 | 0.6263 | 0.4174 | 0.0746 |

Итог по цели эксперимента хороший: `v1.4` почти дошел до `0.50 PR-AUC`, но не перешел этот рубеж. При этом `ROC-AUC` вырос сильнее: до `0.8987`. Это означает, что LightGBM стал лучше ранжировать клиентов в общем смысле и особенно полезен для ROC/FPR-логики.

## Вывод по конфигурациям

Средние test-результаты по конфигурациям показывают, что главным победителем стал блок `lgbm_auc_unweighted`:

| Config | Mean test PR-AUC | Max test PR-AUC | Mean test ROC-AUC | Max test ROC-AUC |
|---|---:|---:|---:|---:|
| `lgbm_auc_unweighted` | 0.4942 | 0.4967 | 0.8985 | 0.8987 |
| `lgbm_pr_default` | 0.4784 | 0.4809 | 0.8659 | 0.8703 |
| `lgbm_pr_regularized` | 0.4704 | 0.4740 | 0.8794 | 0.8850 |
| `lgbm_auc_more_leaves` | 0.4525 | 0.4569 | 0.8910 | 0.8915 |
| `lgbm_auc_default` | 0.4523 | 0.4545 | 0.8922 | 0.8929 |

Самый важный методологический вывод: `class_weight = balanced` не всегда лучший вариант для итогового ранжирования. В `v1.4` unweighted LightGBM с early stopping по `ROC-AUC` оказался сильнее и по `ROC-AUC`, и по `PR-AUC`. Вероятное объяснение: class weighting в предыдущих моделях агрессивно поднимал recall и расширял risk-зону, но ухудшал качество верхнего ранжирования и калибровку score. Unweighted-модель стала более строгой: при threshold `0.5` она помечает только около `1.32%` клиентов как fraud-like, зато precision при этом около `75.3%`.

## Пороги и важный нюанс score scale

Для лучшей модели `lgbm_auc_unweighted_seed2026` threshold `0.5` уже является очень строгим:

| Threshold | Precision | Recall | F1 | Доля клиентов в risk |
|---:|---:|---:|---:|---:|
| 0.50 | 0.7529 | 0.2659 | 0.3930 | 0.0132 |
| 0.60 | 0.8240 | 0.2115 | 0.3366 | 0.0096 |
| 0.70 | 0.8476 | 0.1427 | 0.2443 | 0.0063 |
| 0.80 | 0.9067 | 0.0698 | 0.1296 | 0.0029 |
| 0.90 | 1.0000 | 0.0164 | 0.0323 | 0.0006 |

Это отличается от прошлых weighted LightGBM/CatBoost моделей, где threshold около `0.70-0.80` был рабочей зоной баланса. Для unweighted-модели полезные пороги ниже: например, `Recall@FPR=3%` достигается примерно при threshold около `0.136`, а `Recall@FPR=5%` - примерно при threshold около `0.091`. Поэтому для этой модели нельзя использовать старую threshold-сетку как финальную. Нужно отдельно добавить более низкие thresholds: `0.05`, `0.10`, `0.15`, `0.20`, `0.30`, `0.40`.

## ROC/FPR operating points

Операционные ROC/FPR-метрики подтверждают, что лучшая модель хорошо работает в банковском режиме ограничения ложных срабатываний:

| FPR target | Best model | Actual FPR | Recall@FPR | Precision@FPR | Threshold |
|---:|---|---:|---:|---:|---:|
| 0.5% | `lgbm_pr_default_seed42` | 0.00497 | 0.3244 | 0.7166 | 0.8823 |
| 1% | `lgbm_auc_unweighted_seed2026` | 0.00997 | 0.4148 | 0.6168 | 0.3049 |
| 2% | `lgbm_auc_unweighted_seed777` | 0.01955 | 0.5144 | 0.5045 | 0.1829 |
| 3% | `lgbm_auc_unweighted_seed42` | 0.03000 | 0.5647 | 0.4215 | 0.1349 |
| 5% | `lgbm_auc_unweighted_seed2026` | 0.04998 | 0.6437 | 0.3326 | 0.0913 |

Практический смысл: если банк готов ошибочно затронуть около `1%` нормальных клиентов, модель ловит примерно `41.5%` fraud-like клиентов. Если допустимый FPR поднять до `3%`, recall становится около `56.5%`. Если допустимый FPR поднять до `5%`, recall становится около `64.4%`.

Это полезная интерпретация для диплома, потому что она переводит модель из абстрактного `PR-AUC` в банковский вопрос: сколько fraud-like клиентов можно поймать при контролируемом воздействии на нормальных клиентов.

## Top-k анализ

Top-k качество также осталось сильным. Для `lgbm_auc_unweighted_seed2026` на test:

| Top rate | Precision@k | Recall@k | Fraud found |
|---:|---:|---:|---:|
| Top 1% | 0.8130 | 0.2187 | 213 |
| Top 3% | 0.5643 | 0.4548 | 443 |
| Top 5% | 0.4182 | 0.5616 | 547 |
| Top 10% | 0.2620 | 0.7033 | 685 |
| Top 15% | 0.1897 | 0.7639 | 744 |

Лучший top 1% среди всех запусков был у `lgbm_pr_default_seed777`: `precision@top1% = 0.8244`, найдено 216 fraud-like клиентов. Но по общей `PR-AUC`, `ROC-AUC`, top 3%, top 5%, top 10% и `Recall@FPR` устойчивее выглядит `lgbm_auc_unweighted`.

## Top-5000 client export

Выгрузки top-5000 были созданы:

```text
ml_v1_4_top5000_clients_by_pr_auc_model.csv
ml_v1_4_top5000_test_clients_by_pr_auc_model.csv
ml_v1_4_top5000_clients_by_roc_auc_model.csv
ml_v1_4_top5000_test_clients_by_roc_auc_model.csv
```

В данном запуске лучшая модель по `val PR-AUC` и лучшая модель по `val ROC-AUC` совпали: `lgbm_auc_unweighted_seed2026`. Поэтому PR- и ROC-версии top-5000 используют одну и ту же модель.

Важно правильно читать fraud-rate в этих файлах. В `top5000 all` fraud-rate равен `0.7362`, но этот список включает train, val и test, то есть содержит клиентов, на которых модель обучалась или подбиралась. Его нельзя использовать как честную метрику качества. Он полезен как рабочий ranking для дальнейшего анализа. В `top5000 test` fraud-rate равен `0.157`. Это ниже, потому что это только отложенная часть, и потому что top-5000 из test - это примерно 19% test-выборки, а не верхние 1-5%. При базовой fraud-доле около 3.7% значение 15.7% все равно означает заметный uplift.

## Feature importance

У лучшей модели среди самых важных признаков:

```text
tx_amount_mean
tx_amount_median_proxy
id_19_mode
id_20_mode
id_02_median
id_02_mean
tx_amount_sum
avg_inter_tx_seconds
tx_amount_std
tx_dt_span_days
id_06_mean
id_01_mean
num_missing_identity
crypto_pattern_score
id_13_mode
id_05_mean
odd_amount_share
tx_sum_high_risk_flow_proxy
tx_sum_stanadart_flow_proxy_proxy
non_null_id_values
tx_freq_per_day
high_risk_vs_mean
```

Это согласуется с более ранними выводами: модель опирается на суммы, activity/time, identity-блок и risk/flow proxy признаки. Также подтверждается вывод `v1.2`: identity-блок остается важным источником сигнала.

## Итог по v1.4

`v1.4` успешен. Он не перешел через `0.50 PR-AUC`, но дал новый лучший результат: test `PR-AUC = 0.4967` и `ROC-AUC = 0.8987`. Это лучший результат проекта на текущей витрине без нового feature engineering.

Основной кандидат после `v1.4`: `lgbm_auc_unweighted_seed2026`.

Главная методологическая находка: unweighted LightGBM с early stopping по `ROC-AUC` лучше, чем weighted LightGBM, если оценивать модель как ранжирующий risk-score инструмент. Для финального использования нужно не брать threshold `0.5` по умолчанию, а выбирать режим по бизнес-ограничению: top-k, `Recall@FPR=1%`, `Recall@FPR=3%`, `Recall@FPR=5%` или отдельный precision/recall threshold.

Следующий реальный шанс на скачок выше `0.50 PR-AUC` - не еще один простой перебор моделей, а новый feature engineering. Для `v1.5` наиболее перспективно добавить graph/linkage признаки через card/email/device/identity связи, а также отдельные temporal burst и smoothed ratio признаки для клиентов с короткой историей.
