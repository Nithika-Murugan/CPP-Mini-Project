from __future__ import annotations

import re
from pathlib import Path
from statistics import median

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "enhanced_coconut_prediction_outputs"
PREDICTION_FILES = ("FINAL_TEST_PREDICTIONS.csv", "enhanced_test_predictions.csv")
COMPARISON_FILES = ("FINAL_TEST_RESULTS.csv", "enhanced_model_comparison.csv")
FEATURE_IMPORTANCE_FILES = ("FEATURE_IMPORTANCE.csv", "SHAP_feature_importance.csv")

MODEL_COLUMN_MAP = {
    "Weighted Hybrid": "Weighted_Hybrid",
    "Weighted_Hybrid": "Weighted_Hybrid",
    "Stacking": "Stacking",
    "STL + ML": "STL_ML",
    "STL_ML": "STL_ML",
    "Tuned CatBoost": "Tuned_CatBoost",
    "Tuned_CatBoost": "Tuned_CatBoost",
    "Tuned XGBoost": "Tuned_XGBoost",
    "Tuned_XGBoost": "Tuned_XGBoost",
    "Tuned ExtraTrees": "Tuned_ExtraTrees",
    "Tuned_ExtraTrees": "Tuned_ExtraTrees",
    "Enhanced Tuned CatBoost": "Tuned_CatBoost",
    "Enhanced Tuned XGBoost": "Tuned_XGBoost",
    "Enhanced Tuned ExtraTrees": "Tuned_ExtraTrees",
    "Enhanced Validation-Weighted Hybrid": "Weighted_Hybrid",
    "Enhanced Hybrid": "Weighted_Hybrid",
    "Enhanced_Hybrid": "Weighted_Hybrid",
}


def pick_first_existing(directory: Path, *candidates: str) -> Path | None:
    for candidate in candidates:
        path = directory / candidate
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"₹{float(value):,.2f}"


def money_range(low: float | int | None, high: float | int | None) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "N/A"
    return f"₹{float(low):,.0f} – ₹{float(high):,.0f}"


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "Modal_Price": "Modal_Price",
        "Weighted_Hybrid": "Weighted_Hybrid",
        "Weighted Hybrid": "Weighted_Hybrid",
        "Tuned_CatBoost": "Tuned_CatBoost",
        "Tuned CatBoost": "Tuned_CatBoost",
        "Tuned_XGBoost": "Tuned_XGBoost",
        "Tuned XGBoost": "Tuned_XGBoost",
        "Tuned_ExtraTrees": "Tuned_ExtraTrees",
        "Tuned ExtraTrees": "Tuned_ExtraTrees",
    }
    for old_name, new_name in alias_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    return df


def choose_best_prediction_column(pred_df: pd.DataFrame, model_df: pd.DataFrame | None) -> str:
    if model_df is not None and {"Model", "RMSE"}.issubset(model_df.columns):
        ranked = model_df.sort_values(["RMSE", "MAE"], ascending=True, na_position="last").reset_index(drop=True)
        model_name = ranked.iloc[0].get("Model")
        if model_name is not None:
            mapped = MODEL_COLUMN_MAP.get(str(model_name), None)
            if mapped and mapped in pred_df.columns:
                return mapped

    for candidate in [
        "Weighted_Hybrid",
        "Tuned_CatBoost",
        "Tuned_XGBoost",
        "Tuned_ExtraTrees",
        "STL_ML",
        "Stacking",
    ]:
        if candidate in pred_df.columns:
            return candidate
    for column in pred_df.columns:
        if column.lower().endswith("price") or "hybrid" in column.lower() or "catboost" in column.lower() or "xgboost" in column.lower() or "extratrees" in column.lower():
            return column
    return pred_df.columns[0] if len(pred_df.columns) else ""


def humanize_feature(raw_name: str) -> str:
    name = str(raw_name).strip()
    if not name:
        return "Market signal"

    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    translations = {
        "MarketLag 1d": "Price yesterday",
        "MarketLag 7d": "Price 7 days ago",
        "MarketLag 30d": "Price 30 days ago",
        "MarketLag 90d": "Price 90 days ago",
        "MarketRollMean 7d": "Average market price over the last 7 days",
        "MarketRollMean 30d": "Average market price over the last 30 days",
        "MarketRollMean 90d": "Average market price over the last 90 days",
        "MarketMomentum7": "Recent price trend",
        "MarketMomentum30": "30-day price trend",
        "RegionalIndexLag 1d": "Regional market price",
        "RegionalIndexLag 7d": "Regional market trend",
        "T2M": "Temperature",
        "PRECTOTCORR": "Rainfall",
        "ALLSKY SFC SW DWN": "Sunlight",
        "Month": "Seasonal pattern",
        "Quarter": "Seasonal period",
        "MVLag 1d": "Nearby market trend",
        "RainSum 7d": "Rain in the last 7 days",
        "RainSum 30d": "Rain in the last 30 days",
        "RainSum 90d": "Rain in the last 90 days",
    }

    if name in translations:
        return translations[name]

    if "PriorMean" in name:
        days = re.search(r"(\d+)d", name)
        if days:
            return f"Average price over the last {days.group(1)} days"
        return "Recent average price"

    if "Lag" in name or "lag" in name:
        return "Recent market history"

    if "Momentum" in name:
        return "Recent price movement"

    if "RollMean" in name or "Mean" in name:
        return "Recent average market movement"

    if "CCF" in name:
        return "Market relationship signal"

    if "Rain" in name:
        return "Recent rainfall signal"

    if "Temp" in name.upper() or "T2M" in name.upper():
        return "Temperature signal"

    if "Solar" in name or "ALLSKY" in name.upper():
        return "Sunlight signal"

    return name.title()


def explain_features(feature_df: pd.DataFrame) -> list[tuple[str, str]]:
    if feature_df.empty:
        return []

    top = feature_df.head(5).copy()
    explanations: list[tuple[str, str]] = []
    for _, row in top.iterrows():
        feature = str(row.iloc[0]) if not row.empty else "Market signal"
        label = humanize_feature(feature)
        if any(token in feature for token in ["Lag", "Momentum", "RollMean", "PriorMean"]):
            detail = "This signal is among the strongest market indicators in the current estimate."
        elif any(token in feature for token in ["Rain", "PRECTOTCORR", "CCF"]):
            detail = "This recent weather or market signal is influencing the current estimate."
        elif "Month" in feature or "Quarter" in feature:
            detail = "This seasonal pattern is contributing to the current price estimate."
        else:
            detail = "This factor is part of the current market signal affecting the estimate."
        explanations.append((label, detail))
    return explanations


def result_outlook(predicted_value: float | int | None, history_values: pd.Series) -> tuple[str, str, str]:
    if predicted_value is None or history_values.empty:
        return "MEDIUM", "Prices are expected to remain around the current level.", "WATCH"

    if len(history_values) < 3:
        return "MEDIUM", "Not enough recent comparison data to judge the trend clearly.", "WATCH"

    recent_median = float(median(history_values.dropna().tolist()))
    if predicted_value > recent_median * 1.03:
        return "HIGH", "Prices are expected to remain strong.", "SELL NOW"
    if predicted_value < recent_median * 0.97:
        return "LOW", "Prices are expected to be weaker.", "WAIT"
    return "MEDIUM", "Prices are expected to remain around the current level.", "WATCH"


def build_recommendation_text(outlook: str, predicted_value: float | int | None, recent_median: float | int | None) -> str:
    if outlook == "HIGH":
        if predicted_value is not None and recent_median is not None:
            return "Prices are expected to be stronger than recent levels, so selling now may be reasonable."
        return "The current estimate is stronger than nearby market signals, so selling now may be reasonable."
    if outlook == "LOW":
        if predicted_value is not None and recent_median is not None:
            return "Prices are expected to soften relative to recent levels, so waiting may give a better price."
        return "The current estimate is softer than nearby market signals, so waiting may give a better price."
    return "Prices are expected to stay close to recent levels, so watching the market before selling may be prudent."


def find_prediction_row(pred_df: pd.DataFrame, market: str, variety: str, chosen_date: pd.Timestamp) -> tuple[pd.Series | None, str | None, pd.DataFrame]:
    if pred_df.empty:
        return None, "Prediction data is not available yet. Please run the prediction pipeline first.", pred_df

    filtered = pred_df.copy()
    if "Market" in filtered.columns:
        filtered = filtered[filtered["Market"].astype(str).str.strip() == str(market).strip()].copy()
    if "Variety" in filtered.columns:
        filtered = filtered[filtered["Variety"].astype(str).str.strip() == str(variety).strip()].copy()

    if filtered.empty:
        return None, "Prediction data is not available for this market and variety yet.", filtered

    if "Date" in filtered.columns:
        filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
        filtered = filtered.dropna(subset=["Date"]).sort_values("Date").copy()

    if filtered.empty:
        return None, "Prediction data is not available for this market and variety yet.", filtered

    exact_match = filtered[filtered["Date"] == pd.Timestamp(chosen_date)]
    if not exact_match.empty:
        return exact_match.iloc[0], None, filtered

    nearest_index = (filtered["Date"] - pd.Timestamp(chosen_date)).abs().idxmin()
    nearest_date = filtered.loc[nearest_index, "Date"]
    nearest_label = nearest_date.strftime("%d %b %Y")
    return None, f"Prediction is not available for this date yet. The nearest available date is {nearest_label}.", filtered


st.set_page_config(page_title="Coconut Price Advisor", page_icon="🥥", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
        .hero-title { font-size: 2.8rem; font-weight: 800; line-height: 1.1; color: #1f3d1a; margin-bottom: 0.15rem; }
        .hero-sub { font-size: 1.15rem; color: #4e5b4d; margin-bottom: 1rem; }
        .card { background: #ffffff; border: 1px solid #e7ecf0; border-radius: 18px; padding: 1rem 1.15rem; box-shadow: 0 4px 12px rgba(0,0,0,0.04); }
        .price-box { background: linear-gradient(135deg, #ebf9ee, #f7fdf8); border: 1px solid #cfead5; border-radius: 20px; padding: 1.2rem; }
        .price-number { font-size: clamp(2.2rem, 4vw, 4.2rem); font-weight: 800; line-height: 1; color: #18662d; }
        .tag { display:inline-block; border-radius: 999px; padding: 0.4rem 0.85rem; font-weight:700; font-size:0.92rem; }
        .tag-high { background:#eafbe6; color: #1c7a2b; }
        .tag-medium { background:#fff4d6; color: #8a6100; }
        .tag-low { background:#fde9e8; color: #aa2d2d; }
        .tag-sell { background:#eafbe6; color: #1c7a2b; }
        .tag-wait { background:#fff4d6; color: #8a6100; }
        .tag-watch { background:#e8f0ff; color: #2857af; }
        .small-note { color: #5c6770; font-size: 0.9rem; }
        .factor-item { padding: 0.75rem 0.9rem; border-radius: 14px; border: 1px solid #edf1f3; background: #fafcfb; margin-bottom: 0.6rem; }
        .factor-label { font-weight: 700; color: #1d2a20; }
        .factor-text { color: #48545f; margin-top: 0.15rem; }
        label[data-testid="stBaseButton-secondary"] { font-weight: 700; }
        @media (max-width: 768px) { .hero-title { font-size: 2.1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


if not OUTPUT_DIR.exists():
    st.markdown('<div class="hero-title">🥥 Coconut Price Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Know the expected coconut price before you sell.</div>', unsafe_allow_html=True)
    st.warning("Prediction data is not available yet. Please run the prediction pipeline first.")
    st.stop()

prediction_path = pick_first_existing(OUTPUT_DIR, *PREDICTION_FILES)
comparison_path = pick_first_existing(OUTPUT_DIR, *COMPARISON_FILES)
feature_importance_path = pick_first_existing(OUTPUT_DIR, *FEATURE_IMPORTANCE_FILES)

if prediction_path is None:
    st.markdown('<div class="hero-title">🥥 Coconut Price Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Know the expected coconut price before you sell.</div>', unsafe_allow_html=True)
    st.warning("Prediction data is not available yet. Please run the prediction pipeline first.")
    st.stop()

pred_df = normalize_prediction_columns(load_csv(prediction_path))
model_df = load_csv(comparison_path) if comparison_path is not None else pd.DataFrame()
feature_df = load_csv(feature_importance_path) if feature_importance_path is not None else pd.DataFrame()

if "Date" in pred_df.columns:
    pred_df["Date"] = pd.to_datetime(pred_df["Date"], errors="coerce")
    pred_df = pred_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

best_prediction_column = choose_best_prediction_column(pred_df, model_df)

st.markdown('<div class="hero-title">🥥 Coconut Price Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Know the expected coconut price before you sell.</div>', unsafe_allow_html=True)

markets = []
if "Market" in pred_df.columns:
    markets = sorted(pred_df["Market"].dropna().astype(str).str.strip().unique().tolist())
if not markets:
    st.error("Prediction data is not available yet. Please run the prediction pipeline first.")
    st.stop()

varieties = []
if "Variety" in pred_df.columns:
    varieties = sorted(pred_df["Variety"].dropna().astype(str).str.strip().unique().tolist())
if not varieties:
    varieties = ["Other"]

available_dates = pd.DatetimeIndex(pd.to_datetime(pred_df["Date"].dropna().unique())).sort_values().tolist() if "Date" in pred_df.columns else []

with st.form("advisor_form"):
    market = st.selectbox("Select Market", options=markets, index=0)
    variety = st.selectbox("Select Variety", options=varieties, index=0)
    selected_date = st.date_input(
        "Select Selling Date",
        min_value=(min(available_dates).date() if available_dates else pd.Timestamp.today().date()),
        max_value=(max(available_dates).date() if available_dates else pd.Timestamp.today().date()),
        value=(available_dates[-1].date() if available_dates else pd.Timestamp.today().date()),
    )
    submitted = st.form_submit_button("CHECK EXPECTED PRICE", use_container_width=True)

if submitted:
    row, not_available_message, history_df = find_prediction_row(pred_df, market, variety, pd.Timestamp(selected_date))

    if row is None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.warning(not_available_message)
        if not history_df.empty:
            nearest_dates = history_df["Date"].dropna().sort_values()
            if not nearest_dates.empty:
                nearest = nearest_dates.iloc[(nearest_dates - pd.Timestamp(selected_date)).abs().argmin()]
                st.caption(f"Nearest available date: {nearest.strftime('%d %b %Y')}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    selected_prediction = row.get(best_prediction_column)
    if selected_prediction is None or pd.isna(selected_prediction):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.warning("Prediction is not available for this date yet.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    market_history = history_df[history_df["Market"].astype(str).str.strip() == str(market).strip()]
    market_history = market_history[market_history["Variety"].astype(str).str.strip() == str(variety).strip()].copy()
    if "Date" in market_history.columns:
        market_history["Date"] = pd.to_datetime(market_history["Date"], errors="coerce")
        market_history = market_history.dropna(subset=["Date"]).sort_values("Date")

    range_low = None
    range_high = None
    if not market_history.empty:
        recent_window = market_history.copy()
        if "Date" in recent_window.columns:
            recent_window = recent_window[recent_window["Date"].between(pd.Timestamp(selected_date) - pd.Timedelta(days=30), pd.Timestamp(selected_date) + pd.Timedelta(days=30))]
        predicted_values = recent_window[best_prediction_column].dropna()
        if not predicted_values.empty:
            range_low = float(predicted_values.min())
            range_high = float(predicted_values.max())

    outlook_label, outlook_text, action_label = result_outlook(float(selected_prediction), market_history[best_prediction_column].dropna() if not market_history.empty else pd.Series(dtype=float))
    recommendation_text = build_recommendation_text(outlook_label, float(selected_prediction), float(median(market_history[best_prediction_column].dropna().tolist())) if not market_history.empty and not market_history[best_prediction_column].dropna().empty else None)

    st.markdown('<div class="price-box">', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Expected Price</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="price-number">{money(float(selected_prediction))} / quintal</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-note">Likely range: {money_range(range_low, range_high)}</div>', unsafe_allow_html=True)
    if outlook_label == "HIGH":
        st.markdown('<div class="tag tag-high">📈 HIGH</div>', unsafe_allow_html=True)
    elif outlook_label == "LOW":
        st.markdown('<div class="tag tag-low">📉 LOW</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="tag tag-medium">📊 MEDIUM</div>', unsafe_allow_html=True)
    if outlook_text:
        st.markdown(f'<div class="small-note" style="margin-top: 0.6rem;">{outlook_text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 1.55rem; font-weight: 800; color: #1d2a20; margin-bottom: 0.7rem;">Why this price?</div>', unsafe_allow_html=True)

    if not feature_df.empty:
        factors = explain_features(feature_df)
        if factors:
            for label, detail in factors:
                st.markdown(f'<div class="factor-item"><div class="factor-label">🟢 {label}</div><div class="factor-text">{detail}</div></div>', unsafe_allow_html=True)
        else:
            st.info("The available market data is limited, so a detailed factor breakdown is not available yet.")
    else:
        st.info("The available market data is limited, so a detailed factor breakdown is not available yet.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 1.55rem; font-weight: 800; color: #1d2a20; margin-bottom: 0.7rem;">What should I do?</div>', unsafe_allow_html=True)
    if action_label == "SELL NOW":
        st.markdown('<div class="tag tag-sell">🟢 SELL NOW</div>', unsafe_allow_html=True)
    elif action_label == "WAIT":
        st.markdown('<div class="tag tag-wait">🟡 WAIT</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="tag tag-watch">🔵 WATCH</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-note" style="margin-top: 0.75rem;">{recommendation_text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="small-note" style="margin-top: 1rem;">Price predictions are estimates and actual market prices may vary.</div>', unsafe_allow_html=True)
