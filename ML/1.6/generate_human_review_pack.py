from __future__ import annotations

import argparse
import html
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_CSV = ROOT / "client_profile_v1_0_shuffled.csv"
DEFAULT_HIGH_RISK_CSV = (
    ROOT / "ML" / "1.4" / "results_lightgbm_confirm" / "ml_v1_4_top5000_test_clients_by_roc_auc_model.csv"
)
DEFAULT_SCORE_MODEL_FILE = (
    ROOT / "ML" / "1.4" / "results_lightgbm_confirm" / "ml_v1_4_best_by_roc_auc.joblib"
)
DEFAULT_OUTPUT_DIR = ROOT / "ML" / "1.6" / "results_human_review"


VISIBLE_COLUMNS = [
    "client_id",
    "tx_count_total",
    "tx_amount_sum",
    "tx_amount_mean",
    "tx_amount_std",
    "tx_freq_per_day",
    "daily_activity_share",
    "avg_inter_tx_seconds",
    "short_turnover_share",
    "amount_repeat_share",
    "odd_amount_share",
    "cash_out_ratio_proxy",
    "MCC_risk_share_proxy",
    "high_risk_vs_mean",
    "crypto_pattern_score",
    "low_history_flag",
    "history_support_score",
    "productcd_nunique",
    "addr2_nunique",
    "card4_mode",
    "card6_mode",
    "tx_dt_span_days",
    "top_email_domain",
    "identity_present",
    "num_missing_identity",
    "identity_rows",
    "non_null_id_values",
    "device_type_nunique",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сформировать обезличенный пакет профилей для экспертной оценки."
    )
    parser.add_argument("--profile-csv", type=Path, default=DEFAULT_PROFILE_CSV)
    parser.add_argument("--high-risk-csv", type=Path, default=DEFAULT_HIGH_RISK_CSV)
    parser.add_argument("--score-model-file", type=Path, default=DEFAULT_SCORE_MODEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--n-realistic",
        type=int,
        default=50,
        help="Number of cases in experiment 1: calm realistic-flow block.",
    )
    parser.add_argument(
        "--realistic-fraud-rate",
        type=float,
        default=0.037,
        help="Ожидаемая доля fraud в спокойных и случайных блоках.",
    )
    parser.add_argument(
        "--n-realistic-high-risk-fraud",
        type=int,
        default=5,
        help="Number of high-score true-fraud cases inserted into experiment 1.",
    )
    parser.add_argument(
        "--realistic-high-risk-min-score",
        type=float,
        default=0.8,
        help="Minimum model score for high-risk fraud insertions in experiment 1.",
    )
    parser.add_argument(
        "--realistic-high-risk-skip-top-score",
        type=int,
        default=3,
        help="Skip the top-N highest-score fraud cases for experiment 1 insertions to avoid thin one-operation cases.",
    )
    parser.add_argument(
        "--n-high-risk",
        type=int,
        default=17,
        help="Number of true-fraud high model-score profiles in experiment 2.",
    )
    parser.add_argument(
        "--n-control",
        type=int,
        default=17,
        help="Number of middle-score profiles: the intended manual-control / limit zone.",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=17,
        help="Количество случайных спокойных профилей во втором эксперименте.",
    )
    parser.add_argument(
        "--control-min-score",
        type=float,
        default=0.5,
        help="Lower fraud-score bound for the manual-control bucket.",
    )
    parser.add_argument(
        "--control-max-score",
        type=float,
        default=0.7,
        help="Upper fraud-score bound for the manual-control bucket.",
    )
    parser.add_argument(
        "--control-target-score",
        type=float,
        default=0.6,
        help="Fallback target score if the requested control range has too few profiles.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-pool",
        choices=["full_base", "exclude_top5000"],
        default="exclude_top5000",
        help="Источник для спокойных и случайных блоков.",
    )
    return parser.parse_args()


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def _validate_requested_size(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _sample_exact(df: pd.DataFrame, n: int, seed: int, label: str) -> pd.DataFrame:
    if n == 0:
        return _empty_like(df)
    if len(df) < n:
        raise ValueError(f"Not enough rows for {label}: requested {n}, доступно {len(df)}")
    return df.sample(n=n, random_state=seed).copy()


def _score_сводка(df: pd.DataFrame, score_col: str = "fraud_risk_score") -> dict[str, float | None]:
    scores = pd.to_numeric(df.get(score_col), errors="coerce").dropna()
    if scores.empty:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(scores.min()),
        "median": float(scores.median()),
        "max": float(scores.max()),
    }


RUSSIAN_COLUMN_NAMES = {
    "case_id": "Код кейса",
    "experiment_block": "Код эксперимента",
    "experiment_name": "Эксперимент",
    "case_order": "Порядок внутри эксперимента",
    "sample_bucket": "Служебная корзина",
    "client_id": "client_id",
    "fraud_risk_score": "Fraud-score модели",
    "profile_fraud_label": "Истинная метка fraud",
    "split": "Выборка модели",
    "model_name": "Название модели",
    "source_row_index": "Индекс исходной строки",
    "tx_count_total": "Количество операций",
    "tx_amount_sum": "Совокупный оборот, $",
    "tx_amount_mean": "Средний чек, $",
    "tx_amount_std": "Разброс сумм, $",
    "tx_freq_per_day": "Интенсивность операций",
    "daily_activity_share": "Доля активных дней",
    "avg_inter_tx_seconds": "Средняя пауза между операциями, сек.",
    "short_turnover_share": "Доля коротких пауз",
    "amount_repeat_share": "Доля повторяющихся сумм",
    "odd_amount_share": "Доля неровных сумм",
    "cash_out_ratio_proxy": "Доля дебетового контура",
    "MCC_risk_share_proxy": "Доля оборота в риск-каналах",
    "high_risk_vs_mean": "Объём риск-каналов в средних чеках",
    "crypto_pattern_score": "Композитный сигнал риск-каналов и сумм",
    "low_history_flag": "Флаг короткой истории",
    "history_support_score": "Надёжность истории",
    "productcd_nunique": "Количество платёжных сценариев",
    "addr2_nunique": "Количество региональных признаков addr2",
    "card4_mode": "Основная платёжная система карты",
    "card6_mode": "Основной тип карты",
    "tx_dt_span_days": "Длительность следа, дней",
    "top_email_domain": "Почтовый домен",
    "identity_present": "Есть технический след",
    "num_missing_identity": "Количество пропущенных identity-полей",
    "identity_rows": "Количество identity-записей",
    "non_null_id_values": "Заполненные identity-значения",
    "device_type_nunique": "Количество типов устройств",
    "profile_text": "Описание профиля",
}


EXPERIMENT_1_CODE = "experiment_1_realistic_flow"
EXPERIMENT_1_NAME = "Эксперимент 1: спокойный рабочий поток"
EXPERIMENT_2_CODE = "experiment_2_enriched_risk_zones"
EXPERIMENT_2_NAME = "Эксперимент 2: обогащённая проверка риск-зон"


def _to_russian_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: RUSSIAN_COLUMN_NAMES.get(col, col) for col in df.columns})


def _build_blinded_export(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Эксперимент": df["experiment_name"].tolist(),
            "Порядковый номер": df["case_order"].tolist(),
            "Код кейса": df["case_id"].tolist(),
            "Описание профиля": df["profile_text"].tolist(),
        }
    )


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    if "profile_fraud_label" not in df.columns:
        return {}
    labels = pd.to_numeric(df["profile_fraud_label"], errors="coerce").fillna(-1).astype(int)
    return {str(k): int(v) for k, v in labels.value_counts().sort_index().to_dict().items()}


def _sample_with_base_fraud_rate(
    df: pd.DataFrame,
    n: int,
    доля fraud: float,
    seed: int,
    label: str,
) -> pd.DataFrame:
    if n == 0:
        return _empty_like(df)
    if "profile_fraud_label" not in df.columns:
        raise ValueError("Для стратифицированной выборки в profile_df нужна колонка profile_fraud_label")
    if not 0 <= fraud_rate <= 1:
        raise ValueError(f"fraud_rate must be between 0 and 1, got {fraud_rate}")

    labels = pd.to_numeric(df["profile_fraud_label"], errors="coerce").fillna(0).astype(int)
    fraud_pool = df[labels == 1]
    nonfraud_pool = df[labels == 0]
    n_fraud = int(round(n * fraud_rate))
    if n > 0 and fraud_rate > 0 and n_fraud == 0:
        n_fraud = 1
    n_fraud = min(n_fraud, n)
    n_nonfraud = n - n_fraud

    if len(fraud_pool) < n_fraud:
        raise ValueError(f"Not enough fraud rows for {label}: requested {n_fraud}, доступно {len(fraud_pool)}")
    if len(nonfraud_pool) < n_nonfraud:
        raise ValueError(
            f"Not enough non-fraud rows for {label}: requested {n_nonfraud}, доступно {len(nonfraud_pool)}"
        )

    sampled_parts = []
    if n_fraud:
        sampled_parts.append(fraud_pool.sample(n=n_fraud, random_state=seed + 1))
    if n_nonfraud:
        sampled_parts.append(nonfraud_pool.sample(n=n_nonfraud, random_state=seed + 2))
    sampled = pd.concat(sampled_parts, ignore_index=False).sample(frac=1, random_state=seed + 3)
    return sampled.copy()


def select_human_obvious_high_risk_fraud(
    high_risk_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    n: int,
    min_score: float,
    skip_top_score: int,
) -> pd.DataFrame:
    if n == 0:
        return _empty_like(high_risk_df)

    candidates = high_risk_df.copy()
    if "profile_fraud_label" in candidates.columns:
        candidates = candidates[candidates["profile_fraud_label"] == 1].copy()
    candidates = candidates[candidates["fraud_risk_score"] >= min_score].copy()
    candidates = candidates.sort_values("fraud_risk_score", ascending=False).reset_index(drop=True)
    if skip_top_score > 0:
        candidates = candidates.iloc[skip_top_score:].copy()

    profile_cols = [
        "client_id",
        "tx_count_total",
        "tx_amount_sum",
        "short_turnover_share",
        "cash_out_ratio_proxy",
        "MCC_risk_share_proxy",
        "high_risk_vs_mean",
    ]
    доступно_profile_cols = [col for col in profile_cols if col in profile_df.columns]
    scored = candidates.merge(
        profile_df[доступно_profile_cols],
        on="client_id",
        how="left",
        suffixes=("", "_profile_for_obviousness"),
    )

    def numeric_series(name: str) -> pd.Series:
        if name not in scored.columns:
            return pd.Series(0.0, index=scored.index)
        return pd.to_numeric(scored[name], errors="coerce").fillna(0.0)

    amount_rank = numeric_series("tx_amount_sum").rank(pct=True).fillna(0.0)
    scored["_human_obvious_score"] = (
        pd.to_numeric(scored["fraud_risk_score"], errors="coerce").fillna(0.0) * 2.0
        + numeric_series("tx_count_total").clip(0, 10) / 10.0
        + numeric_series("short_turnover_share").clip(0, 1)
        + numeric_series("cash_out_ratio_proxy").clip(0, 1)
        + numeric_series("MCC_risk_share_proxy").clip(0, 1)
        + numeric_series("high_risk_vs_mean").clip(0, 5) / 5.0
        + amount_rank * 0.3
    )

    picked_ids = (
        scored.sort_values(["_human_obvious_score", "fraud_risk_score"], ascending=[False, False])
        .head(n)["client_id"]
        .tolist()
    )
    if len(picked_ids) < n:
        raise ValueError(
            f"Недостаточно очевидных high-score fraud-профилей: запрошено {n}, доступно {len(picked_ids)}"
        )

    return (
        high_risk_df[high_risk_df["client_id"].isin(picked_ids)]
        .assign(_pick_order=lambda df: df["client_id"].map({client_id: i for i, client_id in enumerate(picked_ids)}))
        .sort_values("_pick_order")
        .drop(columns=["_pick_order"], errors="ignore")
        .copy()
    )


def score_profiles_with_lgbm_bundle(
    profile_df: pd.DataFrame,
    client_ids: list[str],
    model_file: Path,
) -> tuple[pd.Series, str]:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "Для расчета Fraud-score модели при сборке пакета экспертной оценки нужен joblib."
        ) from exc

    try:
        bundle = joblib.load(model_file)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not load the LightGBM bundle. Install compatible dependencies first: "
            "lightgbm and scikit-learn==1.6.1."
        ) from exc

    required_keys = {"model", "preprocessor", "features"}
    missing_keys = required_keys - set(bundle.keys())
    if missing_keys:
        raise ValueError(f"Score model bundle is missing keys: {sorted(missing_keys)}")

    features = list(bundle["features"])
    missing_features = [col for col in features if col not in profile_df.columns]
    if missing_features:
        raise ValueError(
            "Profile CSV does not contain LightGBM features: "
            + ", ".join(missing_features[:20])
        )

    unique_ids = pd.Index(pd.Series(client_ids, dtype="string").dropna().unique())
    scoring_df = (
        profile_df[profile_df["client_id"].isin(unique_ids)]
        .drop_duplicates("client_id", keep="first")
        .set_index("client_id")
    )
    missing_clients = [client_id for client_id in unique_ids if client_id not in scoring_df.index]
    if missing_clients:
        raise ValueError(
            "Could not find profile rows for client_id values: "
            + ", ".join(map(str, missing_clients[:20]))
        )

    X = scoring_df.loc[unique_ids, features].copy()
    X_encoded = bundle["preprocessor"].transform(X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        scores = bundle["model"].predict_proba(X_encoded)[:, 1]
    return pd.Series(scores, index=unique_ids, name="fraud_risk_score"), str(
        bundle.get("model_name", "lgbm_auc_unweighted_seed2026")
    )


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _fmt_num(value: object, digits: int = 1) -> str:
    number = _safe_float(value)
    if number is None:
        return "нет данных"
    return f"{number:.{digits}f}".replace(".", ",")


def _fmt_pct(value: object) -> str:
    number = _safe_float(value)
    if number is None:
        return "нет данных"
    return f"{number * 100:.1f}%".replace(".", ",")


def _fmt_usd(value: object) -> str:
    number = _safe_float(value)
    if number is None:
        return "нет данных"
    return f"${number:,.2f}".replace(",", " ")


def _plural_ru(number: int, one: str, few: str, many: str) -> str:
    number_abs = abs(number) % 100
    last = number_abs % 10
    if 10 < number_abs < 20:
        return many
    if last == 1:
        return one
    if 1 < last < 5:
        return few
    return many


def _human_duration(seconds: object) -> str:
    value = _safe_float(seconds)
    if value is None or value <= 0:
        return "нет данных"
    if value < 60:
        return f"около {int(round(value))} сек."
    if value < 3600:
        return f"около {int(round(value / 60))} мин."
    if value < 86400:
        return f"около {_fmt_num(value / 3600, 1)} ч."
    return f"около {_fmt_num(value / 86400, 1)} дн."


def _band_from_quantiles(value: object, series: pd.Series, reverse: bool = False) -> str:
    number = _safe_float(value)
    if number is None:
        return "нет данных"
    sample = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if sample.empty:
        return "нет данных"
    q10, q25, q75, q90 = sample.quantile([0.10, 0.25, 0.75, 0.90]).tolist()
    if reverse:
        if number <= q10:
            return "очень высокий"
        if number <= q25:
            return "высокий"
        if number <= q75:
            return "умеренный"
        if number <= q90:
            return "низкий"
        return "очень низкий"
    if number <= q10:
        return "очень низкий"
    if number <= q25:
        return "низкий"
    if number <= q75:
        return "умеренный"
    if number <= q90:
        return "высокий"
    return "очень высокий"


def _share_band(value: object, reverse: bool = False) -> str:
    number = _safe_float(value)
    if number is None:
        return "нет данных"
    thresholds = [0.05, 0.20, 0.50, 0.80]
    labels = ["очень низкая", "низкая", "умеренная", "высокая", "очень высокая"]
    if reverse:
        labels = list(reversed(labels))
    if number <= thresholds[0]:
        return labels[0]
    if number <= thresholds[1]:
        return labels[1]
    if number <= thresholds[2]:
        return labels[2]
    if number <= thresholds[3]:
        return labels[3]
    return labels[4]


def build_quantile_context(df: pd.DataFrame) -> dict[str, pd.Series]:
    ctx: dict[str, pd.Series] = {}
    for col in [
        "tx_count_total",
        "tx_amount_sum",
        "tx_amount_mean",
        "tx_amount_std",
        "tx_freq_per_day",
        "avg_inter_tx_seconds",
        "identity_rows",
        "non_null_id_values",
        "device_type_nunique",
    ]:
        if col in df.columns:
            ctx[col] = df[col]
    return ctx


def _badge_tone(badge: str) -> str:
    badge = (badge or "").lower()
    if "нет данных" in badge or "нет домена" in badge:
        return "unknown"
    if (
        "очень высок" in badge
        or "высок" in badge
        or "выше среднего" in badge
        or "много чеков" in badge
        or "странная почта" in badge
    ):
        return "high"
    if (
        "умерен" in badge
        or "средн" in badge
        or "достаточ" in badge
        or "несколько" in badge
        or "редкий домен" in badge
    ):
        return "medium"
    if (
        "очень низ" in badge
        or "низ" in badge
        or "слаб" in badge
        or "корот" in badge
        or "один" in badge
        or "нет заметного" in badge
        or "обычная почта" in badge
    ):
        return "low"
    return "neutral"


def _item(label: str, badge: str, сводка: str, help_text: str) -> dict[str, str]:
    return {
        "label": label,
        "badge": badge,
        "сводка": сводка,
        "help": help_text,
        "tone": _badge_tone(badge),
    }


def _history_support_badge(score: float | None) -> str:
    if score is None:
        return "нет данных"
    if score < 0.2:
        return "очень короткая"
    if score < 0.5:
        return "ограниченная"
    if score < 1.0:
        return "достаточная"
    return "устойчивая"


def _history_support_сводка(score: float | None) -> str:
    if score is None:
        return "По длине истории пока нельзя уверенно судить, насколько устойчивы остальные сигналы."
    if score < 0.2:
        return "Данных по клиенту пока очень мало, поэтому любые выводы стоит считать предварительными."
    if score < 0.5:
        return "Базовый рисунок поведения уже виден, но профиль ещё не выглядит полностью устойчивым."
    if score < 1.0:
        return "Истории уже хватает для анализа, хотя запас наблюдений ещё не максимальный."
    return "По клиенту накоплено достаточно наблюдений, чтобы воспринимать профиль как устойчивый."


def _history_volume_сводка(tx_count: int | None) -> str:
    if tx_count is None:
        return "По количеству операций профиль оценить не удалось."
    op_text = _plural_ru(tx_count, "операция", "операции", "операций")
    if tx_count <= 1:
        return f"В профиле только {tx_count} {op_text}. Это почти пустая история: по ней видно конкретный эпизод, но ещё плохо видно обычное поведение клиента."
    if tx_count <= 3:
        return f"В профиле {tx_count} {op_text}. История короткая: уже можно увидеть первые признаки поведения, но выводы лучше считать осторожными."
    if tx_count <= 9:
        return f"В профиле {tx_count} {op_text}. История ограниченная, но в ней уже есть несколько точек для сравнения темпа, сумм и платёжных сценариев."
    return f"В профиле {tx_count} {op_text}. История достаточно насыщенная: по ней можно увереннее судить о повторяемом поведении клиента."


def _high_risk_checks_badge(value: float | None) -> str:
    if value is None:
        return "нет данных"
    if value <= 0.05:
        return "нет заметного объёма"
    if value < 1.0:
        return "меньше одного чека"
    if value < 2.0:
        return "около одного чека"
    if value < 5.0:
        return "несколько чеков"
    return "много чеков"


def _high_risk_checks_сводка(value: float | None) -> str:
    if value is None:
        return "Не удалось оценить, какой объём прошёл через более рискованные платёжные каналы."
    if value <= 0.05:
        return "Через более рискованные каналы почти не было оборота относительно обычного размера операции клиента."
    if value < 1.0:
        return f"Через более рискованные каналы прошёл объём меньше одного обычного чека клиента: примерно {value:.1f} среднего чека."
    if value < 2.0:
        return f"Через более рискованные каналы прошёл объём примерно как один обычный чек клиента: около {value:.1f} среднего чека."
    if value < 5.0:
        return f"Через более рискованные каналы прошёл объём примерно как несколько обычных операций клиента: около {value:.1f} средних чеков."
    return f"Через более рискованные каналы прошёл крупный для этого клиента объём: около {value:.1f} средних чеков."


def _frequency_сводка(band: str) -> str:
    if "нет данных" in band:
        return "Оценить интенсивность операций по истории не удалось."
    if "очень высокий" in band or "высок" in band:
        return "Относительно остальных профилей в выборке клиент проводит операции заметно чаще. Такой темп выглядит плотнее обычного клиентского поведения."
    if "умерен" in band:
        return "Интенсивность операций близка к средней части выборки: профиль не выглядит ни явно редким, ни чрезмерно плотным по темпу."
    return "Относительно остальных профилей в выборке клиент проводит операции редко. По одному этому признаку темп не выглядит настораживающе плотным."


def _gap_сводка(band: str) -> str:
    if "нет данных" in band:
        return "Для оценки пауз между операциями данных пока не хватает."
    if "очень высокий" in band or "высок" in band:
        return "Паузы между операциями короче, чем у большинства профилей в выборке. Операции идут плотнее и ближе друг к другу по времени."
    if "умерен" in band:
        return "Паузы между операциями находятся примерно в средней зоне выборки: профиль не выглядит ни явно растянутым, ни резко ускоренным."
    return "Паузы между операциями длиннее, чем у большинства профилей в выборке. Операции выглядят более разнесёнными по времени."


COMMON_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "icloud.com",
    "mail.com",
    "aol.com",
    "comcast.net",
    "verizon.net",
    "live.com",
    "msn.com",
}


def _email_domain_badge(value: object) -> str:
    if _is_missing(value):
        return "нет домена"
    domain = str(value).strip().lower()
    if domain in {"nan", "none", "unknown", "missing"}:
        return "нет домена"
    if domain in COMMON_EMAIL_DOMAINS:
        return "обычная почта"
    if "." not in domain or len(domain) < 5:
        return "странная почта"
    return "редкий домен"


def _email_domain_сводка(value: object) -> str:
    if _is_missing(value):
        return "Email-домен в профиле не выделился. Для ручной проверки это означает, что по почте нет нормальной опоры для сопоставления."
    domain = str(value).strip().lower()
    if domain in {"nan", "none", "unknown", "missing"}:
        return "Email-домен выглядит отсутствующим или нераспознанным. Это не доказывает риск, но снижает качество контекста по клиенту."
    if domain in COMMON_EMAIL_DOMAINS:
        return f"Основной домен почты — {domain}. Это распространённый публичный почтовый домен, сам по себе он не выглядит странно."
    if "." not in domain or len(domain) < 5:
        return f"Основной домен почты выглядит нестандартно: {domain}. Такой домен стоит воспринимать как дополнительный повод внимательнее посмотреть профиль."
    return f"Основной домен почты — {domain}. Это не один из самых массовых публичных доменов, поэтому его лучше воспринимать как дополнительный контекст для проверки."


def build_case_sections(row: pd.Series, quantiles: dict[str, pd.Series]) -> list[dict[str, object]]:
    tx_count = _safe_float(row.get("tx_count_total"))
    span_days = _safe_float(row.get("tx_dt_span_days"))
    history_support = _safe_float(row.get("history_support_score"))
    freq = row.get("tx_freq_per_day")
    mean_gap = row.get("avg_inter_tx_seconds")
    short_share = row.get("short_turnover_share")
    daily_share = row.get("daily_activity_share")
    amount_sum = row.get("tx_amount_sum")
    amount_mean = row.get("tx_amount_mean")
    amount_std = row.get("tx_amount_std")
    repeat_share = row.get("amount_repeat_share")
    odd_share = row.get("odd_amount_share")
    cash_out = row.get("cash_out_ratio_proxy")
    mcc_proxy = row.get("MCC_risk_share_proxy")
    high_risk_vs_mean = row.get("high_risk_vs_mean")
    crypto_score = row.get("crypto_pattern_score")
    identity_present = int(_safe_float(row.get("identity_present")) or 0)
    identity_rows = row.get("identity_rows")
    non_null_ids = row.get("non_null_id_values")
    devices = row.get("device_type_nunique")
    email = row.get("top_email_domain")
    card4 = row.get("card4_mode")
    card6 = row.get("card6_mode")
    productcd_nunique = row.get("productcd_nunique")

    history_band = _band_from_quantiles(tx_count, quantiles.get("tx_count_total", pd.Series(dtype=float)))
    freq_band = _band_from_quantiles(freq, quantiles.get("tx_freq_per_day", pd.Series(dtype=float)))
    amount_band = _band_from_quantiles(amount_sum, quantiles.get("tx_amount_sum", pd.Series(dtype=float)))
    mean_amount_band = _band_from_quantiles(amount_mean, quantiles.get("tx_amount_mean", pd.Series(dtype=float)))
    volatility_band = _band_from_quantiles(amount_std, quantiles.get("tx_amount_std", pd.Series(dtype=float)))
    gap_band = _band_from_quantiles(
        mean_gap, quantiles.get("avg_inter_tx_seconds", pd.Series(dtype=float)), reverse=True
    )
    identity_rows_band = _band_from_quantiles(
        identity_rows, quantiles.get("identity_rows", pd.Series(dtype=float))
    )
    non_null_ids_band = _band_from_quantiles(
        non_null_ids, quantiles.get("non_null_id_values", pd.Series(dtype=float))
    )
    devices_band = _band_from_quantiles(
        devices, quantiles.get("device_type_nunique", pd.Series(dtype=float))
    )

    tx_count_int = int(tx_count) if tx_count is not None else None
    productcd_int = int(float(productcd_nunique)) if not _is_missing(productcd_nunique) else None
    devices_int = int(float(devices)) if not _is_missing(devices) else None

    if span_days is None:
        span_badge = "нет данных"
        span_сводка = "По длительности следа операций сделать вывод не удалось."
    elif span_days <= 0:
        span_badge = "однодневный след"
        span_сводка = "Вся наблюдаемая активность уложилась в один день или меньше."
    elif span_days <= 3:
        span_badge = "короткий след"
        span_сводка = f"Операции растянуты примерно на {_fmt_num(span_days, 1)} дня, профиль пока короткий."
    elif span_days <= 14:
        span_badge = "несколько дней"
        span_сводка = f"Операции наблюдаются примерно {_fmt_num(span_days, 1)} дня, базовый ритм уже виден."
    else:
        span_badge = "длинный след"
        span_сводка = f"Операции растянуты примерно на {_fmt_num(span_days, 1)} дня, у профиля есть временная глубина."

    short_band = _share_band(short_share)
    if short_band in {"очень низкая", "низкая"}:
        short_сводка = f"Плотные серии встречаются редко: {_fmt_pct(short_share)}. Деньги не выглядят как поток, который постоянно проходит почти без пауз."
    elif short_band == "умеренная":
        short_сводка = f"Часть операций идёт плотными сериями: {_fmt_pct(short_share)}. Это уже заметно быстрее спокойного клиентского поведения."
    else:
        short_сводка = f"Плотные серии встречаются часто: {_fmt_pct(short_share)}. Профиль похож на быстрое прокручивание средств."

    daily_share_value = _safe_float(daily_share)
    if daily_share_value is None:
        daily_badge = "нет данных"
        daily_сводка = "Оценить распределение активности по дням не удалось."
    elif daily_share_value >= 0.8:
        daily_badge = "равномерно по дням"
        daily_сводка = f"Активность замечена в {_fmt_pct(daily_share)} дней наблюдаемого периода. Поведение распределено по времени, а не собрано в короткий всплеск."
    elif daily_share_value >= 0.4:
        daily_badge = "смешанный ритм"
        daily_сводка = f"Активность замечена в {_fmt_pct(daily_share)} дней наблюдаемого периода. Есть и спокойные дни, и заметные всплески."
    else:
        daily_badge = "всплесками"
        daily_сводка = f"Активность замечена только в {_fmt_pct(daily_share)} дней наблюдаемого периода. Профиль выглядит более эпизодическим и всплесковым."

    repeat_band = _share_band(repeat_share)
    if repeat_band in {"очень низкая", "низкая"}:
        repeat_сводка = f"Одинаковые суммы почти не повторяются: {_fmt_pct(repeat_share)}. Поведение не выглядит шаблонным по суммам."
    elif repeat_band == "умеренная":
        repeat_сводка = f"Одинаковые суммы встречаются заметно: {_fmt_pct(repeat_share)}. В профиле уже есть элементы повторяемого паттерна."
    else:
        repeat_сводка = f"Много операций проходит на одинаковые суммы: {_fmt_pct(repeat_share)}. Это похоже на регулярный или полуавтоматический сценарий."

    odd_band = _share_band(odd_share)
    if odd_band in {"очень низкая", "низкая"}:
        odd_сводка = f"Неровные суммы почти не встречаются: {_fmt_pct(odd_share)}. Большинство операций выглядит привычно по форме суммы."
    elif odd_band == "умеренная":
        odd_сводка = f"Неровные суммы встречаются периодически: {_fmt_pct(odd_share)}. Это уже заметный, но не экстремальный сигнал."
    else:
        odd_сводка = f"Неровных сумм много: {_fmt_pct(odd_share)}. Суммы часто выглядят нетипично для спокойного клиентского профиля."

    cash_band = _share_band(cash_out)
    if cash_band in {"очень низкая", "низкая"}:
        cash_сводка = f"Доля операций по дебетовым картам среди операций с понятным карточным контуром низкая: {_fmt_pct(cash_out)}. По этому признаку профиль не выглядит как активное расходование/перевод средств дальше."
    elif cash_band == "умеренная":
        cash_сводка = f"Дебетовый контур заметен: {_fmt_pct(cash_out)} среди операций с понятным карточным контуром. Это может быть нормальным расходованием, но в сочетании с быстрым темпом и риск-каналами требует внимания."
    else:
        cash_сводка = f"В профиле сильно преобладают операции по дебетовым картам: {_fmt_pct(cash_out)} среди операций с понятным карточным контуром. Такой рисунок ближе к сценарию активного расходования или перевода средств дальше."

    risk_flow_band = _share_band(mcc_proxy)
    if risk_flow_band in {"очень низкая", "низкая"}:
        risk_flow_сводка = f"Через более рискованные платёжные каналы проходит малая часть оборота: {_fmt_pct(mcc_proxy)}. В обучающей базе каналы C/H/S в сумме давали около 9,1% fraud против 2,2% у остальных, но у этого клиента такой сигнал выражен слабо."
    elif risk_flow_band == "умеренная":
        risk_flow_сводка = f"Через более рискованные платёжные каналы проходит заметная часть оборота: {_fmt_pct(mcc_proxy)}. В обучающей базе каналы C/H/S встречались у fraud заметно чаще, особенно канал C, поэтому такой рисунок уже требует внимания."
    else:
        risk_flow_сводка = f"Значительная часть оборота идёт через более рискованные платёжные каналы: {_fmt_pct(mcc_proxy)}. В обучающей базе набор C/H/S был связан с fraud заметно чаще, чем остальные каналы, поэтому это сильный сигнал на ручную проверку."

    crypto_band = _share_band(crypto_score)
    if crypto_band in {"очень низкая", "низкая"}:
        crypto_сводка = "Сочетание рискованных каналов и заметных сумм выражено слабо: профиль не похож на быстрый прогон ощутимых денег."
    elif crypto_band == "умеренная":
        crypto_сводка = "Есть умеренное сочетание: часть оборота идёт через более рискованные каналы, и средний чек уже не выглядит мелким."
    else:
        crypto_сводка = "Одновременно выражены и более рискованные каналы, и заметный средний чек. Для ручной проверки это выглядит как сценарий быстрого прогона ощутимых сумм."

    if productcd_int is None:
        product_сводка = "Информация о платёжном сценарии в профиль не попала."
        product_badge = "нет данных"
    elif productcd_int <= 1:
        product_сводка = "В наблюдаемой истории клиент действует в одном и том же платёжном сценарии."
        product_badge = "один сценарий"
    elif productcd_int == 2:
        product_сводка = "Клиент переключается между двумя платёжными сценариями."
        product_badge = "два сценария"
    else:
        product_сводка = f"Клиент использует несколько платёжных сценариев: минимум {productcd_int}."
        product_badge = "несколько сценариев"

    if not identity_present:
        session_badge = "данных мало"
        session_сводка = "По клиенту почти нет технической информации об устройстве, браузере и онлайн-сеансе."
    elif identity_rows_band in {"очень низкий", "низкий"}:
        session_badge = "след короткий"
        session_сводка = "Технический след присутствует, но деталей о среде устройства и сеанса немного."
    elif identity_rows_band == "умеренный":
        session_badge = "след заметный"
        session_сводка = "По клиенту есть нормальный объём технических данных об устройстве и онлайн-сеансе."
    else:
        session_badge = "след насыщенный"
        session_сводка = "По клиенту собралось много технических данных: устройство и сеанс видны достаточно подробно."

    sections: list[dict[str, object]] = [
        {
            "title": "История профиля",
            "items": [
                _item(
                    "Объём наблюдаемой истории",
                    history_band,
                    _history_volume_сводка(tx_count_int),
                    "Показывает, сколько операций попало в профиль. Чем короче история, тем меньше опоры для вывода: один эпизод может быть случайностью, а длинная история лучше показывает устойчивый рисунок поведения.",
                ),
                _item(
                    "Длительность следа операций",
                    span_badge,
                    span_сводка,
                    "Это не дата регистрации клиента, а длина наблюдаемого временного следа по операциям внутри профиля.",
                ),
                _item(
                    "Надёжность выводов по профилю",
                    _history_support_badge(history_support),
                    _history_support_сводка(history_support),
                    "Служебный показатель того, хватает ли наблюдений, чтобы считать профиль устойчивым. Это не риск сам по себе, а качество опоры для вывода.",
                ),
            ],
        },
        {
            "title": "Темп и ритм",
            "items": [
                _item(
                    "Интенсивность операций",
                    freq_band,
                    _frequency_сводка(freq_band),
                    "Сравнение идёт не с абсолютной нормой, а с другими клиентскими профилями в этой выборке. Высокая интенсивность означает, что операции идут чаще, чем у большинства клиентов; низкая - что профиль более редкий по активности.",
                ),
                _item(
                    "Паузы между операциями",
                    gap_band,
                    _gap_сводка(gap_band),
                    "Считается как количество операций поделённое на время активности профиля. Сравнение идёт с паузами в других профилях выборки. Короткие паузы означают, что операции идут почти подряд и могут быть похожи на быстрый прогон средств; длинные паузы выглядят спокойнее по ритму.",
                ),
                _item(
                    "Плотные серии операций",
                    short_band,
                    short_сводка,
                    "Показывает, как часто операции образуют быстрые цепочки (т.е. отношение маленьких пауз между операциями и большими паузами, а не просто среднее время между операциями). Для аналитика это прокси на быстрое движение денег без естественных пауз.",
                ),
                _item(
                    "Распределение активности по дням",
                    daily_badge,
                    daily_сводка,
                    "Если активность размазана по дням, профиль обычно спокойнее. Если она собрана в короткие всплески, это больше похоже на эпизодическую кампанию.",
                ),
            ],
        },
        {
            "title": "Денежный рисунок",
            "items": [
                _item(
                    "Совокупный оборот",
                    amount_band,
                    f"За весь наблюдаемый след по профилю прошло около {_fmt_usd(amount_sum)}. По масштабу это {amount_band} уровень относительно остальных клиентов в выборке.",
                    "Здесь сравнивается общий оборот профиля с другими профилями. Сам по себе большой объём не означает риск, но меняет контекст остальных признаков.",
                ),
                _item(
                    "Типичный размер операции",
                    mean_amount_band,
                    f"Типичный размер одной операции около {_fmt_usd(amount_mean)}. Это {mean_amount_band} чек относительно выборки.",
                    "Это относительный показатель. Он помогает понять, работает ли клиент на мелких, средних или крупных суммах по сравнению с базой.",
                ),
                _item(
                    "Разброс сумм",
                    volatility_band,
                    (
                        f"Оценочный разброс вокруг типичного чека около {_fmt_usd(amount_std)}. Суммы операций в истории в основном похожи друг на друга."
                        if volatility_band in {"очень низкий", "низкий"}
                        else f"Оценочный разброс вокруг типичного чека около {_fmt_usd(amount_std)}. Суммы операций меняются заметно, но без экстремальных скачков."
                        if volatility_band == "умеренный"
                        else f"Оценочный разброс вокруг типичного чека около {_fmt_usd(amount_std)}. Суммы операций сильно скачут от операции к операции."
                    ),
                    "Это сравнение со средним клиентом по базе. Низкий разброс означает однотипные суммы, высокий — что клиент работает на очень разных чеках.",
                ),
                _item(
                    "Повторы одних и тех же сумм",
                    repeat_band,
                    repeat_сводка,
                    "Когда одни и те же суммы повторяются слишком часто, это может быть похоже на шаблонный поток или полумеханический сценарий работы.",
                ),
                _item(
                    "Неровные суммы",
                    odd_band,
                    odd_сводка,
                    "Показатель на суммы, которые выглядят неестественно для спокойной розничной активности: дробные, рваные, нетипично точные. Пример: 46,19 долларов.",
                ),
            ],
        },
        {
            "title": "Сценарий движения средств",
            "items": [
                _item(
                    "Дебетовый контур операций",
                    cash_band,
                    cash_сводка,
                    "В данных нет готового поля «входящий/исходящий платёж», поэтому мы не утверждаем направление денег напрямую. Этот показатель считается проще: debit трактуется как контур расходования средств, credit и часть более спокойных W/R-сценариев — как противоположный карточный контур. Чем выше доля debit, тем больше профиль похож на активное расходование или перевод средств дальше; сам по себе этот признак не означает fraud.",
                ),
                _item(
                    "Операции в более рискованных платёжных каналах",
                    risk_flow_band,
                    risk_flow_сводка,
                    "Здесь мы смотрим, какая доля оборота прошла через каналы ProductCD C, H и S. В обучающей базе их совокупная fraud-доля была около 9,1% против 2,2% у остальных каналов; сильнее всего выделялся канал C. Это не приговор, а статистически более настораживающий маршрут движения денег.",
                ),
                _item(
                    "Объём в риск-каналах относительно обычного чека",
                    _high_risk_checks_badge(_safe_float(high_risk_vs_mean)),
                    _high_risk_checks_сводка(_safe_float(high_risk_vs_mean)),
                    "Считается так: сумма операций клиента в каналах C/H/S делится на средний чек этого же клиента. Например, значение 3 означает, что через более рискованные каналы прошёл объём примерно как три обычные операции клиента. Это помогает понять, насколько риск-каналы значимы именно для масштаба этого профиля.",
                ),
                _item(
                    "Признак быстрого прогона заметных сумм",
                    crypto_band,
                    crypto_сводка,
                    "Сигнал растёт, когда у клиента одновременно заметна доля операций в каналах C/H/S и средний чек не выглядит мелким. Его удобно читать как общий индикатор сценария «быстрый прогон заметных сумм», а не как отдельный тип мошенничества.",
                ),
            ],
        },
        {
            "title": "Технический и продуктовый контекст",
            "items": [
                _item(
                    "Платёжный сценарий клиента",
                    product_badge,
                    product_сводка,
                    "Этот блок опирается на то, какой тип продукта используется, разнообразные ли эти продукты. В итоговой витрине сохранено только количество разных сценариев, а не сами исходные коды, поэтому точный тип продукта здесь не показывается.",
                ),
                _item(
                    "Карточный профиль",
                    "контекст",
                    f"Основная платёжная система: {card4 if not _is_missing(card4) else 'нет данных'}; тип карты: {card6 if not _is_missing(card6) else 'нет данных'}.",
                    "Это справочный контекст по карте, а не отдельный риск-сигнал. Он помогает аналитикам быстрее понять среду клиента.",
                ),
                _item(
                    "След устройства и онлайн-сеанса",
                    session_badge,
                    session_сводка,
                    "Речь о технических данных из identity-слоя: устройство, браузер, ОС, параметры сеанса.",
                ),
                _item(
                    "Смена устройств",
                    devices_band,
                    (
                        f"В истории замечено {devices_int} {_plural_ru(devices_int, 'тип', 'типа', 'типов')} устройства."
                        if devices_int is not None
                        else "Данных о разнообразии устройств нет."
                    ),
                    "Если устройство одно и то же, профиль обычно выглядит стабильнее. Несколько разных устройств повышают вариативность и усложняют ручную проверку.",
                ),
                _item(
                    "Почтовый домен",
                    _email_domain_badge(email),
                    _email_domain_сводка(email),
                    "Смотрим основной email-домен в истории профиля. Распространённые публичные домены вроде gmail/yahoo/hotmail обычно читаются как нейтральный контекст. Редкий, пустой или плохо распознаваемый домен не доказывает fraud, но может быть дополнительным поводом внимательнее посмотреть профиль.",
                ),
            ],
        },
    ]
    return sections


def describe_profile(row: pd.Series, quantiles: dict[str, pd.Series]) -> str:
    sections = build_case_sections(row, quantiles)
    lines = ["Что известно по профилю:"]
    for section in sections:
        lines.append(f"- {section['title']}:")
        for item in section["items"]:
            lines.append(f"  - {item['label']}: {item['badge']}. {item['сводка']}")
    return "\n".join(lines)


def build_case_text(row: pd.Series, quantiles: dict[str, pd.Series]) -> str:
    description = describe_profile(row, quantiles)
    return (
        f"{description}\n\n"
        "Вопрос для эксперта:\n"
        "- Какое решение вы бы приняли: `пропуск`, `на контроль` или `в блок`?\n"
    )


def build_markdown_pack(df: pd.DataFrame, title: str) -> str:
    blocks = [f"# {title}", "", "Ниже приведены обезличенные профили клиентов для экспертной оценки.", ""]
    for _, row in df.iterrows():
        blocks.append(f"## {row['case_id']}")
        blocks.append("")
        blocks.append(row["profile_text"])
        blocks.append("")
        blocks.append("---")
        blocks.append("")
    return "\n".join(blocks)


def build_html_pack(
    df: pd.DataFrame,
    title: str,
    quantiles: dict[str, pd.Series],
    lead_text: str | None = None,
) -> str:
    cards = []
    current_experiment = None
    for idx, row in df.iterrows():
        experiment_name = str(row.get("experiment_name", "")).strip()
        if experiment_name and experiment_name != current_experiment:
            current_experiment = experiment_name
            if row.get("experiment_block") == EXPERIMENT_1_CODE:
                experiment_note = (
                    "Спокойный блок: большинство профилей обычные, доля fraud близка к рабочему потоку. "
                    "Оценивайте как обычную очередь мониторинга."
                )
            elif row.get("experiment_block") == EXPERIMENT_2_CODE:
                experiment_note = (
                    "Обогащённый блок: здесь специально собраны разные риск-зоны модели. "
                    "Он нужен не для оценки реальной частоты блокировок, а для проверки различимости профилей."
                )
            else:
                experiment_note = "Следующий блок карточек относится к отдельной части проверки."
            cards.append(
                f"""
                <section class="experiment-divider">
                  <h2>{html.escape(experiment_name)}</h2>
                  <p>{html.escape(experiment_note)}</p>
                </section>
                """
            )

        section_html_parts: list[str] = []
        for section in build_case_sections(row, quantiles):
            items_html = []
            for item in section["items"]:
                items_html.append(
                    f"""
                    <article class="signal signal--{item['tone']}">
                      <div class="signal-head">
                        <div class="signal-title">{html.escape(str(item['label']))}</div>
                        <div class="signal-head-right">
                          <span class="badge badge--{item['tone']}">{html.escape(str(item['badge']))}</span>
                          <span class="hint" tabindex="0" data-tip="{html.escape(str(item['help']))}">?</span>
                        </div>
                      </div>
                      <div class="signal-сводка">{html.escape(str(item['сводка']))}</div>
                    </article>
                    """
                )
            section_html_parts.append(
                f"""
                <section class="section-card">
                  <h3>{html.escape(str(section['title']))}</h3>
                  <div class="signal-grid">
                    {''.join(items_html)}
                  </div>
                </section>
                """
            )
        cards.append(
            f"""
            <details class="case-card" {'open' if idx == 0 else ''}>
              <сводка>
                <div class="case-title">{html.escape(str(row['case_id']))}</div>
                <div class="case-subtitle">Откройте карточку и выберите: пропуск / на контроль / в блок</div>
              </сводка>
              <div class="case-content">
                {''.join(section_html_parts)}
                <section class="decision-box">
                  <div class="decision-title">Вопрос для эксперта</div>
                  <div class="decision-text">Какое решение вы бы приняли по этому профилю: <strong>пропуск</strong>, <strong>на контроль</strong> или <strong>в блок</strong>?</div>
                </section>
              </div>
            </details>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: Inter, Segoe UI, Arial, sans-serif;
      line-height: 1.55;
      margin: 28px auto;
      max-width: 1180px;
      color: #172033;
      background:
        radial-gradient(circle at top left, rgba(226, 232, 240, 0.85), transparent 28%),
        linear-gradient(180deg, #f7f8fb 0%, #ffffff 100%);
      padding: 0 18px 48px;
    }}
    h1 {{
      font-size: 2rem;
      margin-bottom: 0.35rem;
    }}
    .lead {{
      color: #42526b;
      max-width: 820px;
      margin-bottom: 18px;
    }}
    .legend {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }}
    .legend-item {{
      background: rgba(255,255,255,0.88);
      border: 1px solid #dde4f0;
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 0.95rem;
      color: #42526b;
      box-shadow: 0 10px 24px rgba(38, 54, 78, 0.06);
    }}
    .experiment-divider {{
      margin: 26px 0 14px;
      padding: 18px 20px;
      border-radius: 20px;
      background: linear-gradient(135deg, #172033 0%, #28415f 100%);
      color: #ffffff;
      box-shadow: 0 16px 34px rgba(23, 32, 51, 0.16);
    }}
    .experiment-divider h2 {{
      margin: 0 0 6px 0;
      font-size: 1.28rem;
    }}
    .experiment-divider p {{
      margin: 0;
      color: #dbe7f6;
      max-width: 860px;
    }}
    .case-card {{
      background: rgba(255,255,255,0.95);
      border: 1px solid #dbe4f1;
      border-radius: 20px;
      margin: 16px 0;
      box-shadow: 0 12px 32px rgba(29, 41, 57, 0.08);
      overflow: hidden;
    }}
    .case-card > сводка {{
      list-style: none;
      cursor: pointer;
      padding: 18px 22px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
    }}
    .case-card > сводка::-webkit-details-marker {{
      display: none;
    }}
    .case-title {{
      font-size: 1.12rem;
      font-weight: 700;
      color: #172033;
    }}
    .case-subtitle {{
      color: #5b6b84;
      font-size: 0.93rem;
      text-align: right;
    }}
    .case-content {{
      padding: 20px 22px 24px;
    }}
    .section-card {{
      background: #f9fbff;
      border: 1px solid #e1e9f5;
      border-radius: 18px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .section-card h3 {{
      margin: 0 0 12px 0;
      font-size: 1rem;
      color: #20304a;
    }}
    .signal-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
      gap: 12px;
    }}
    .signal {{
      background: white;
      border: 1px solid #e4ebf7;
      border-radius: 16px;
      padding: 14px 14px 12px;
      box-shadow: 0 8px 20px rgba(31, 46, 72, 0.04);
    }}
    .signal-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .signal-title {{
      font-weight: 700;
      color: #1d2a40;
      font-size: 0.98rem;
    }}
    .signal-head-right {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }}
    .signal-сводка {{
      color: #44546d;
      font-size: 0.94rem;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.79rem;
      font-weight: 700;
      border: 1px solid transparent;
      white-space: nowrap;
    }}
    .badge--high {{
      background: #fff0eb;
      color: #b54708;
      border-color: #fed7c3;
    }}
    .badge--medium {{
      background: #fff8e6;
      color: #b26b00;
      border-color: #f4ddb0;
    }}
    .badge--low {{
      background: #edf7f1;
      color: #1d7a46;
      border-color: #c8e7d3;
    }}
    .badge--unknown, .badge--neutral {{
      background: #eef2f7;
      color: #5c6a7d;
      border-color: #d8e0ea;
    }}
    .hint {{
      position: relative;
      width: 22px;
      height: 22px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      background: #e8eef8;
      color: #1f4f9d;
      font-weight: 800;
      font-size: 0.78rem;
      cursor: help;
      user-select: none;
    }}
    .hint:hover::after,
    .hint:focus::after {{
      content: attr(data-tip);
      position: absolute;
      top: 28px;
      right: 0;
      width: 320px;
      max-width: 68vw;
      background: #132238;
      color: #ffffff;
      border-radius: 14px;
      padding: 12px 14px;
      line-height: 1.45;
      white-space: normal;
      box-shadow: 0 16px 36px rgba(10, 20, 35, 0.28);
      z-index: 40;
      text-align: left;
      font-weight: 500;
    }}
    .decision-box {{
      background: linear-gradient(135deg, #1f4f9d 0%, #285ea8 100%);
      color: white;
      border-radius: 18px;
      padding: 16px 18px;
      margin-top: 6px;
    }}
    .decision-title {{
      font-weight: 800;
      margin-bottom: 6px;
      font-size: 0.95rem;
    }}
    .decision-text {{
      font-size: 0.95rem;
    }}
    @media (max-width: 760px) {{
      .case-card > сводка {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .case-subtitle {{
        text-align: left;
      }}
      .hint:hover::after,
      .hint:focus::after {{
        right: auto;
        left: -140px;
      }}
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="lead">{html.escape(lead_text or 'Обезличенные профили для экспертной оценки. Карточки сгруппированы по смысловым блокам, а справа от каждого сигнала есть подсказка с пояснением, что именно означает показатель и как его читать.')}</p>
  <section class="legend">
    <div class="legend-item"><strong>Как читать карточку:</strong> сначала смотрите краткий вывод по сигналу, затем при необходимости наводите курсор на <strong>?</strong> справа.</div>
    <div class="legend-item"><strong>Важно:</strong> уровни вроде «низкий», «умеренный» и «высокий» — это сравнение с другими клиентскими профилями в выборке, а не абсолютная норма банка.</div>
    <div class="legend-item"><strong>Решение эксперта:</strong> для каждого кейса нужен только один ответ — <strong>пропуск</strong>, <strong>на контроль</strong> или <strong>в блок</strong>.</div>
  </section>
  {''.join(cards)}
</body>
</html>
"""


def build_answer_template(df: pd.DataFrame, pack_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Название пакета": pack_name,
            "Эксперимент": df["experiment_name"].tolist(),
            "Порядковый номер": df["case_order"].tolist(),
            "Код кейса": df["case_id"].tolist(),
            "Решение эксперта": "",
        }
    )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _validate_requested_size("n_realistic", args.n_realistic)
    _validate_requested_size("n_realistic_high_risk_fraud", args.n_realistic_high_risk_fraud)
    _validate_requested_size("realistic_high_risk_skip_top_score", args.realistic_high_risk_skip_top_score)
    _validate_requested_size("n_high_risk", args.n_high_risk)
    _validate_requested_size("n_control", args.n_control)
    _validate_requested_size("n_random", args.n_random)
    if args.n_realistic_high_risk_fraud > args.n_realistic:
        raise ValueError("n_realistic_high_risk_fraud cannot be greater than n_realistic")
    if args.control_min_score >= args.control_max_score:
        raise ValueError("control_min_score must be lower than control_max_score")

    profile_df = pd.read_csv(args.profile_csv)
    high_risk_df = pd.read_csv(args.high_risk_csv)

    if "client_id" not in profile_df.columns:
        raise ValueError("profile_csv must contain client_id")
    if "profile_fraud_label" not in profile_df.columns:
        raise ValueError("profile_csv must contain profile_fraud_label")
    if "client_id" not in high_risk_df.columns:
        raise ValueError("high_risk_csv must contain client_id")
    if "fraud_risk_score" not in high_risk_df.columns:
        raise ValueError("high_risk_csv must contain fraud_risk_score")

    profile_df["client_id"] = profile_df["client_id"].astype(str)
    profile_df["profile_fraud_label"] = pd.to_numeric(
        profile_df["profile_fraud_label"], errors="coerce"
    ).fillna(0).astype(int)
    profile_df = profile_df.drop_duplicates("client_id", keep="first")

    high_risk_df["client_id"] = high_risk_df["client_id"].astype(str)
    high_risk_df["fraud_risk_score"] = pd.to_numeric(
        high_risk_df["fraud_risk_score"], errors="coerce"
    )
    high_risk_df = high_risk_df.drop_duplicates("client_id", keep="first")
    high_risk_df = high_risk_df.dropna(subset=["fraud_risk_score"]).copy()
    if "profile_fraud_label" in high_risk_df.columns:
        high_risk_df["profile_fraud_label"] = pd.to_numeric(
            high_risk_df["profile_fraud_label"], errors="coerce"
        ).fillna(0).astype(int)
    high_risk_df = high_risk_df.sort_values("fraud_risk_score", ascending=False).reset_index(drop=True)

    top_ids_full = set(high_risk_df["client_id"].tolist())
    baseline_excluded_ids = top_ids_full if args.random_pool == "exclude_top5000" else set()

    realistic_high_risk_pick = select_human_obvious_high_risk_fraud(
        high_risk_df,
        profile_df,
        args.n_realistic_high_risk_fraud,
        args.realistic_high_risk_min_score,
        args.realistic_high_risk_skip_top_score,
    )
    realistic_high_risk_ids = set(realistic_high_risk_pick["client_id"].tolist())

    baseline_pool_df = profile_df[~profile_df["client_id"].isin(baseline_excluded_ids)].copy()
    realistic_baseline_size = args.n_realistic - args.n_realistic_high_risk_fraud
    realistic_baseline_pick = _sample_with_base_fraud_rate(
        baseline_pool_df,
        realistic_baseline_size,
        args.realistic_fraud_rate,
        args.seed + 11,
        "experiment 1 realistic flow",
    )
    realistic_ids = set(realistic_baseline_pick["client_id"].tolist()) | realistic_high_risk_ids

    high_risk_candidates = high_risk_df[~high_risk_df["client_id"].isin(realistic_ids)].copy()
    if "profile_fraud_label" in high_risk_candidates.columns:
        high_risk_candidates = high_risk_candidates[
            high_risk_candidates["profile_fraud_label"] == 1
        ].copy()
    high_risk_pick = high_risk_candidates.head(args.n_high_risk).copy()
    if len(high_risk_pick) < args.n_high_risk:
        raise ValueError(
            f"Not enough high-score true-fraud rows: requested {args.n_high_risk}, доступно {len(high_risk_pick)}"
        )
    high_risk_ids = set(high_risk_pick["client_id"].tolist())

    control_pool = high_risk_df[
        ~high_risk_df["client_id"].isin(realistic_ids | high_risk_ids)
    ].copy()
    control_range_pool = control_pool[
        (control_pool["fraud_risk_score"] >= args.control_min_score)
        & (control_pool["fraud_risk_score"] < args.control_max_score)
    ].copy()
    control_range_candidate_count = int(len(control_range_pool))
    control_used_fallback = control_range_candidate_count < args.n_control
    if control_used_fallback:
        control_pool["_score_distance_to_control_target"] = (
            control_pool["fraud_risk_score"] - args.control_target_score
        ).abs()
        control_pick = (
            control_pool.sort_values(
                ["_score_distance_to_control_target", "fraud_risk_score"],
                ascending=[True, False],
            )
            .head(args.n_control)
            .drop(columns=["_score_distance_to_control_target"], errors="ignore")
            .copy()
        )
    else:
        control_pick = _sample_exact(
            control_range_pool,
            args.n_control,
            args.seed + 101,
            "control score bucket",
        )
        control_pick = control_pick.sort_values("fraud_risk_score", ascending=False).copy()

    selected_model_ids = realistic_ids | high_risk_ids | set(control_pick["client_id"].tolist())

    if args.random_pool == "exclude_top5000":
        random_pool_df = profile_df[
            ~profile_df["client_id"].isin(top_ids_full | selected_model_ids)
        ].copy()
    else:
        random_pool_df = profile_df[~profile_df["client_id"].isin(selected_model_ids)].copy()

    random_pick = _sample_with_base_fraud_rate(
        random_pool_df,
        args.n_random,
        args.realistic_fraud_rate,
        args.seed + 51,
        "experiment 2 random baseline bucket",
    )

    high_risk_joined = high_risk_pick.merge(
        profile_df[VISIBLE_COLUMNS], on="client_id", how="left", suffixes=("", "_profile")
    )
    control_joined = control_pick.merge(
        profile_df[VISIBLE_COLUMNS], on="client_id", how="left", suffixes=("", "_profile")
    )
    realistic_high_risk_joined = realistic_high_risk_pick.merge(
        profile_df[VISIBLE_COLUMNS], on="client_id", how="left", suffixes=("", "_profile")
    )

    meta_cols = [
        "client_id",
        "fraud_risk_score",
        "profile_fraud_label",
        "split",
        "model_name",
        "source_row_index",
    ]
    high_risk_joined = high_risk_joined[
        [col for col in meta_cols if col in high_risk_joined.columns]
        + [col for col in VISIBLE_COLUMNS if col in high_risk_joined.columns and col != "client_id"]
    ].copy()
    control_joined = control_joined[
        [col for col in meta_cols if col in control_joined.columns]
        + [col for col in VISIBLE_COLUMNS if col in control_joined.columns and col != "client_id"]
    ].copy()
    realistic_high_risk_joined = realistic_high_risk_joined[
        [col for col in meta_cols if col in realistic_high_risk_joined.columns]
        + [col for col in VISIBLE_COLUMNS if col in realistic_high_risk_joined.columns and col != "client_id"]
    ].copy()

    def prepare_baseline_rows(pick: pd.DataFrame, split_label: str) -> pd.DataFrame:
        rows = pick.copy()
        rows["_profile_row_index"] = rows.index
        rows["fraud_risk_score"] = np.nan
        rows["model_name"] = "not_shown"
        rows["split"] = split_label
        rows["source_row_index"] = rows["_profile_row_index"]
        rows = rows.drop(columns=["_profile_row_index"], errors="ignore")
        return rows[
            [
                "client_id",
                "fraud_risk_score",
                "profile_fraud_label",
                "split",
                "model_name",
                "source_row_index",
            ]
            + [col for col in VISIBLE_COLUMNS if col in rows.columns and col != "client_id"]
        ].copy()

    realistic_baseline_joined = prepare_baseline_rows(realistic_baseline_pick, "base_realistic")
    random_joined = prepare_baseline_rows(random_pick, "base_random")

    realistic_block = (
        pd.concat(
            [
                realistic_baseline_joined.assign(
                    experiment_block=EXPERIMENT_1_CODE,
                    experiment_name=EXPERIMENT_1_NAME,
                    sample_bucket="realistic_base_rate",
                ),
                realistic_high_risk_joined.assign(
                    experiment_block=EXPERIMENT_1_CODE,
                    experiment_name=EXPERIMENT_1_NAME,
                    sample_bucket="realistic_inserted_high_risk_fraud",
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        .sample(frac=1, random_state=args.seed + 71)
        .reset_index(drop=True)
    )
    enriched_block = pd.concat(
        [
            high_risk_joined.assign(
                experiment_block=EXPERIMENT_2_CODE,
                experiment_name=EXPERIMENT_2_NAME,
                sample_bucket="enriched_true_fraud_high_score",
            ),
            control_joined.assign(
                experiment_block=EXPERIMENT_2_CODE,
                experiment_name=EXPERIMENT_2_NAME,
                sample_bucket="enriched_control_score_0_5_0_7",
            ),
            random_joined.assign(
                experiment_block=EXPERIMENT_2_CODE,
                experiment_name=EXPERIMENT_2_NAME,
                sample_bucket="enriched_random_base_rate",
            ),
        ],
        ignore_index=True,
        sort=False,
    ).sample(frac=1, random_state=args.seed + 72).reset_index(drop=True)

    realistic_block["case_order"] = range(1, len(realistic_block) + 1)
    enriched_block["case_order"] = range(1, len(enriched_block) + 1)
    realistic_block["case_id"] = [f"R{idx:03d}" for idx in realistic_block["case_order"]]
    enriched_block["case_id"] = [f"E{idx:03d}" for idx in enriched_block["case_order"]]

    combined = pd.concat(
        [realistic_block, enriched_block],
        ignore_index=True,
        sort=False,
    )

    review_scores, score_model_name = score_profiles_with_lgbm_bundle(
        profile_df,
        combined["client_id"].astype(str).tolist(),
        args.score_model_file,
    )
    combined["fraud_risk_score"] = combined["client_id"].astype(str).map(review_scores).astype(float)
    combined["model_name"] = score_model_name

    quantiles = build_quantile_context(profile_df)

    combined["profile_text"] = combined.apply(lambda row: build_case_text(row, quantiles), axis=1)

    internal_columns = [
        "case_id",
        "experiment_block",
        "experiment_name",
        "case_order",
        "sample_bucket",
        "client_id",
        "fraud_risk_score",
        "profile_fraud_label",
        "split",
        "model_name",
        "source_row_index",
    ] + [col for col in VISIBLE_COLUMNS if col in combined.columns and col != "client_id"] + ["profile_text"]
    internal_df = combined[internal_columns].copy()

    experiment_1_df = combined[combined["experiment_block"] == EXPERIMENT_1_CODE].copy()
    experiment_2_df = combined[combined["experiment_block"] == EXPERIMENT_2_CODE].copy()
    blinded_df = _build_blinded_export(combined)
    experiment_1_blinded = _build_blinded_export(experiment_1_df)
    experiment_2_blinded = _build_blinded_export(experiment_2_df)

    сводка = {
        "seed": args.seed,
        "profile_csv": str(args.profile_csv),
        "high_risk_csv": str(args.high_risk_csv),
        "score_model_file": str(args.score_model_file),
        "score_model_name": score_model_name,
        "n_total": int(len(combined)),
        "n_experiment_1_realistic": int(len(experiment_1_df)),
        "n_experiment_2_enriched": int(len(experiment_2_df)),
        "realistic_fraud_rate_requested": float(args.realistic_fraud_rate),
        "n_realistic_baseline": int((combined["sample_bucket"] == "realistic_base_rate").sum()),
        "n_realistic_inserted_high_risk_fraud": int(
            (combined["sample_bucket"] == "realistic_inserted_high_risk_fraud").sum()
        ),
        "realistic_high_risk_min_score": float(args.realistic_high_risk_min_score),
        "realistic_high_risk_skip_top_score": int(args.realistic_high_risk_skip_top_score),
        "n_high_risk_true_fraud": int(
            (combined["sample_bucket"] == "enriched_true_fraud_high_score").sum()
        ),
        "n_control": int((combined["sample_bucket"] == "enriched_control_score_0_5_0_7").sum()),
        "n_random_base_rate": int((combined["sample_bucket"] == "enriched_random_base_rate").sum()),
        "control_score_range_requested": [
            float(args.control_min_score),
            float(args.control_max_score),
        ],
        "control_target_score": float(args.control_target_score),
        "control_range_candidate_count": control_range_candidate_count,
        "control_used_fallback": bool(control_used_fallback),
        "label_counts_by_experiment": {
            EXPERIMENT_1_CODE: _label_counts(experiment_1_df),
            EXPERIMENT_2_CODE: _label_counts(experiment_2_df),
        },
        "label_counts_by_bucket": {
            bucket: _label_counts(group)
            for bucket, group in combined.groupby("sample_bucket")
        },
        "score_сводка_by_bucket": {
            bucket: _score_сводка(group)
            for bucket, group in combined.groupby("sample_bucket")
        },
        "n_missing_fraud_risk_score": int(combined["fraud_risk_score"].isna().sum()),
        "high_risk_true_fraud_source_split_counts": combined.loc[
            combined["sample_bucket"] == "enriched_true_fraud_high_score", "split"
        ].fillna("unknown").value_counts().to_dict(),
        "control_source_split_counts": combined.loc[
            combined["sample_bucket"] == "enriched_control_score_0_5_0_7", "split"
        ].fillna("unknown").value_counts().to_dict(),
        "random_pool": args.random_pool,
        "bucket_design": "experiment_1 = mostly calm base-rate flow plus a few high-score true-fraud insertions; experiment_2 = equal-size high-score true fraud, control-score, and base-rate random groups.",
        "recommendation": "Analyze experiment 1 and experiment 2 separately. Experiment 1 approximates normal prevalence; experiment 2 is intentionally enriched and must not be used to estimate production block rates.",
    }

    _to_russian_columns(internal_df).to_csv(args.output_dir / "human_review_master_key.csv", index=False, encoding="utf-8-sig")
    blinded_df.to_csv(args.output_dir / "human_review_cases_blinded.csv", index=False, encoding="utf-8-sig")
    build_answer_template(combined, "Общий пакет").to_csv(
        args.output_dir / "human_review_answers.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "human_review_questionnaire.md").write_text(
        build_markdown_pack(combined, "Human Review Pack: две части эксперимента"),
        encoding="utf-8",
    )
    (args.output_dir / "human_review_questionnaire.html").write_text(
        build_html_pack(
            combined,
            "Единый human-review опросник",
            quantiles,
            "Первая часть имитирует спокойный поток с близкой к реальности долей fraud. Вторая часть намеренно обогащена риск-кейсами и нужна для проверки различимости модельных зон.",
        ),
        encoding="utf-8",
    )
    (args.output_dir / "human_review_experiment_descriptions.md").write_text(
        """# Описание двух human-review экспериментов

## Эксперимент 1: спокойный рабочий поток

Цель: проверить, как эксперт принимает решения в условиях, похожих на обычную очередь мониторинга.

Состав: 50 обезличенных профилей. Основная часть берётся из спокойного baseline-пула с долей fraud около 3,7%, а дополнительно в блок вставляются 5 high-score true-fraud профилей с `fraud_risk_score >= 0,8`. Для вставки пропускаются самые верхние однооперационные top-score кейсы и выбираются более человеко-читаемые fraud-профили: больше операций, быстрые цепочки, риск-каналы, дебетовый контур. Это сделано, чтобы в первом блоке были реальные кандидаты на решение `в блок`, но остальная часть потока оставалась спокойной.

Как интерпретировать: этот блок можно использовать для разговора о естественной строгости экспертов, частоте решений `пропуск / на контроль / в блок` и реакции на несколько явно рискованных кейсов внутри почти спокойного потока. Его нельзя считать полностью случайным рабочим днём, потому что часть fraud-кейсов добавлена специально.

## Эксперимент 2: обогащённая проверка риск-зон

Цель: проверить, различают ли эксперты зоны, которые выделяет модель: явный риск, серая зона контроля и спокойный фон.

Состав: три равные группы по 17 профилей:

- высокорисковые true-fraud профили из верхней части рейтинга модели;
- профили из контрольной зоны `fraud_risk_score` от 0,5 до 0,7;
- случайные базовые профили с долей fraud около реального уровня.

Как интерпретировать: этот блок нельзя использовать для оценки реального процента блокировок в проде, потому что он намеренно обогащён риск-кейсами. Его задача другая: проверить, совпадает ли экспертная логика с модельным ранжированием.

## Главное правило анализа

Эксперимент 1 и эксперимент 2 нужно анализировать отдельно. Первый отвечает на вопрос операционного поведения в спокойном потоке, второй — на вопрос различимости риск-зон.
""",
        encoding="utf-8",
    )
    (args.output_dir / "human_review_generation_сводка.json").write_text(
        json.dumps(сводка, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Пакет экспертной оценки сохранен в:", args.output_dir)
    print("Файл `human_review_master_key.csv` используйте только после завершения экспертной оценки.")


if __name__ == "__main__":
    main()
