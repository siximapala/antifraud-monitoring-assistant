# ML v1.4 - краткий сводка результатов

Лучший запуск: `lgbm_auc_unweighted_seed2026`.

На test:

```text
ROC-AUC = 0.8987
PR-AUC  = 0.4967
precision@0.5 = 0.7529
recall@0.5    = 0.2659
F1@0.5        = 0.3930
```

Это новый лучший результат проекта на текущей витрине без нового feature engineering. По `PR-AUC` модель почти дошла до `0.50`, а по `ROC-AUC` поднялась до `0.8987`. Победа стабильная: конфигурация `lgbm_auc_unweighted` была лучшей в среднем по трем seed и дала лучшие значения по `val PR-AUC`, `val ROC-AUC`, `test PR-AUC` и `test ROC-AUC`.

Главная находка `v1.4`: unweighted LightGBM с early stopping по `ROC-AUC` оказался сильнее weighted LightGBM. Это означает, что `class_weight = balanced` не всегда улучшает итоговый risk-score ranking. Weighted-модели давали больше recall при threshold `0.5`, но unweighted-модель лучше ранжирует клиентов и лучше работает в ROC/FPR-режиме.

Для лучшей модели threshold `0.5` очень строгий: он отправляет в risk только около `1.32%` test-клиентов, зато precision достигает `75.3%`. Для рабочего режима этой модели нельзя переносить старые thresholds `0.70-0.80`; нужно выбирать threshold отдельно. По FPR-метрикам:

```text
Recall@FPR=1% ≈ 0.4148
Recall@FPR=3% ≈ 0.5647
Recall@FPR=5% ≈ 0.6437
```

Top-k test для лучшей модели:

```text
Top 1% precision = 0.8130, fraud found = 213
Top 3% precision = 0.5643, fraud found = 443
Top 5% precision = 0.4182, fraud found = 547
Top 10% precision = 0.2620, fraud found = 685
```

Top-5000 выгрузки созданы. `top5000 all` имеет fraud-rate `0.7362`, но включает train/val/test, поэтому это не честная метрика качества. `top5000 test` имеет fraud-rate `0.157`; это корректнее для оценки, но надо помнить, что top-5000 test - это примерно 19% test-выборки, а не top 1-5%.

Вывод: `v1.4` успешен. Основной кандидат теперь `lgbm_auc_unweighted_seed2026`. Следующий шанс на скачок выше `0.50 PR-AUC` лучше искать не в очередном переборе моделей, а в `v1.5` feature engineering: graph/linkage признаки через card/email/device/identity связи, temporal burst features и smoothed ratio признаки для клиентов с короткой историей.
