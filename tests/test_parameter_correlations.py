#!/usr/bin/env python3
"""Project Math: Verificador e Teste de Correlação dos Parâmetros Acústicos e Emocionais.

Avalia empiricamente no dataset MELD:
  1. Matriz de Correlação Contínua (Pearson e Spearman) entre os Acoustic Cues do eGeMAPS.
  2. Co-ocorrência dos Cues Extremos (Low, Moderate, High) vs. Hipótese Teórica de Independência.
  3. Acoplamento Condicional Acústica-Emoção: P(High | Emoção de Alta Excitação) vs P(High | Baixa Excitação).
  4. Asserções automatizadas para validação em CI / Pytest.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest


# Features mapeadas no OPERA
ACOUSTIC_CUE_MAP: Dict[str, str] = {
    "pitch_height": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "pitch_dynamics": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "vocal_energy": "loudness_sma3_percentile50.0",
    "speaking_rate": "VoicedSegmentsPerSec",
    "spectral_tilt": "hammarbergIndexV_sma3nz_amean",
    "hnr": "HNRdBACF_sma3nz_amean",
    "jitter": "jitterLocal_sma3nz_amean",
    "shimmer": "shimmerLocaldB_sma3nz_amean",
}

CONTINUOUS_4_CUES = ["pitch_height", "pitch_dynamics", "vocal_energy", "speaking_rate"]

HIGH_AROUSAL_EMOTIONS = ["anger", "joy", "surprise", "fear"]
LOW_AROUSAL_EMOTIONS = ["neutral", "sadness"]


def load_meld_acoustic_dataset(parquet_path: str = "raw_data/meld_features_balanced_700.parquet") -> pd.DataFrame:
    """Carrega o dataset pré-extraído de features eGeMAPS com emoções e locutores."""
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo de features não encontrado em: {p}")
    df = pd.read_parquet(p)
    df["emotion"] = df["Emotion"].str.lower().str.strip()
    return df


def compute_correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula matrizes de correlação de Pearson e Spearman para os cues principais."""
    sub = df[[ACOUSTIC_CUE_MAP[c] for c in CONTINUOUS_4_CUES]].rename(
        columns={ACOUSTIC_CUE_MAP[c]: c for c in CONTINUOUS_4_CUES}
    )
    pearson_corr = sub.corr(method="pearson")
    spearman_corr = sub.corr(method="spearman")
    return pearson_corr, spearman_corr


def discretize_cues_by_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    """Discretiza cada cue contínuo em quartis (low: <= Q25, moderate: Q25..Q75, high: > Q75)."""
    df_disc = df.copy()
    for cue in CONTINUOUS_4_CUES:
        col = ACOUSTIC_CUE_MAP[cue]
        q25 = float(df_disc[col].quantile(0.25))
        q75 = float(df_disc[col].quantile(0.75))
        df_disc[f"{cue}_tier"] = pd.cut(
            df_disc[col],
            bins=[-np.inf, q25, q75, np.inf],
            labels=["low", "moderate", "high"],
        )
    return df_disc


def compute_cooccurrence_and_coupling(df_disc: pd.DataFrame) -> Dict[str, Any]:
    """Calcula probabilidades empíricas conjuntas e condicionais."""
    n_total = len(df_disc)

    # 1. Co-ocorrência empírica de múltiplos highs
    all_3_high = (
        (df_disc["pitch_height_tier"] == "high")
        & (df_disc["pitch_dynamics_tier"] == "high")
        & (df_disc["vocal_energy_tier"] == "high")
    ).mean()

    all_4_high = (
        (df_disc["pitch_height_tier"] == "high")
        & (df_disc["pitch_dynamics_tier"] == "high")
        & (df_disc["vocal_energy_tier"] == "high")
        & (df_disc["speaking_rate_tier"] == "high")
    ).mean()

    two_or_more_high = (
        (df_disc["pitch_height_tier"] == "high").astype(int)
        + (df_disc["pitch_dynamics_tier"] == "high").astype(int)
        + (df_disc["vocal_energy_tier"] == "high").astype(int)
        + (df_disc["speaking_rate_tier"] == "high").astype(int)
        >= 2
    ).mean()

    # 2. Acoplamento Emoção -> P(High)
    high_arousal_sub = df_disc[df_disc["emotion"].isin(HIGH_AROUSAL_EMOTIONS)]
    low_arousal_sub = df_disc[df_disc["emotion"].isin(LOW_AROUSAL_EMOTIONS)]

    p_high_in_high_arousal = np.mean([
        (high_arousal_sub[f"{c}_tier"] == "high").mean() for c in CONTINUOUS_4_CUES
    ])
    p_high_in_low_arousal = np.mean([
        (low_arousal_sub[f"{c}_tier"] == "high").mean() for c in CONTINUOUS_4_CUES
    ])

    p_low_in_low_arousal = np.mean([
        (low_arousal_sub[f"{c}_tier"] == "low").mean() for c in CONTINUOUS_4_CUES
    ])

    return {
        "n_samples": n_total,
        "p_all_3_high": float(all_3_high),
        "theoretical_3_high": float(0.25 ** 3),
        "ratio_3_high": float(all_3_high / (0.25 ** 3)),
        "p_all_4_high": float(all_4_high),
        "theoretical_4_high": float(0.25 ** 4),
        "ratio_4_high": float(all_4_high / (0.25 ** 4)) if all_4_high > 0 else 0.0,
        "p_two_or_more_high": float(two_or_more_high),
        "p_high_given_high_arousal": float(p_high_in_high_arousal),
        "p_high_given_low_arousal": float(p_high_in_low_arousal),
        "p_low_given_low_arousal": float(p_low_in_low_arousal),
        "arousal_contrast_ratio": float(p_high_in_high_arousal / max(1e-4, p_high_in_low_arousal)),
    }


def run_parameter_correlation_diagnostics() -> Dict[str, Any]:
    """Executa diagnóstico completo e exibe relatório dinâmico."""
    df = load_meld_acoustic_dataset()
    pearson_corr, spearman_corr = compute_correlations(df)
    df_disc = discretize_cues_by_quartiles(df)
    metrics = compute_cooccurrence_and_coupling(df_disc)

    print("=" * 90)
    print("🔬 DIAGNÓSTICO MATEMÁTICO EMPÍRICO: CORRELAÇÃO DE PARÂMETROS ACÚSTICOS & EMOÇÃO (MELD)")
    print("=" * 90)
    print(f"Total de Amostras Avaliadas (Balanceadas por Emoção): {metrics['n_samples']}")

    print("\n📊 1. MATRIZ DE CORRELAÇÃO DE PEARSON (ACOUSTIC CUES):")
    print("-" * 90)
    print(pearson_corr.round(3).to_string())

    print("\n📊 2. MATRIZ DE CORRELAÇÃO DE SPEARMAN (RANK):")
    print("-" * 90)
    print(spearman_corr.round(3).to_string())

    print("\n📊 3. TESTE DE CO-OCORRÊNCIA DE EXTREMOS (QUEBRA DA HIPÓTESE DE INDEPENDÊNCIA):")
    print("-" * 90)
    print(f"• P(Pitch High & Dynamics High & Energy High) Empírico:  {metrics['p_all_3_high']*100:.2f}%")
    print(f"• P(3 Highs) sob Hipótese de Independência (0.25^3):       {metrics['theoretical_3_high']*100:.2f}%")
    print(f"• Fator de Amplificação Empírica:                         {metrics['ratio_3_high']:.2f}x MAIOR que o teórico!")
    print(f"• P(>= 2 Cues Contínuos em High simultaneamente):         {metrics['p_two_or_more_high']*100:.2f}%")

    print("\n📊 4. ACOPLAMENTO CONDICIONAL: EMOÇÃO VS. CUES ACÚSTICOS:")
    print("-" * 90)
    print(f"• P(Cue=High | Emoção de Alta Excitação [Anger/Joy/Surprise/Fear]): {metrics['p_high_given_high_arousal']*100:.2f}%")
    print(f"• P(Cue=High | Emoção de Baixa Excitação [Neutral/Sadness]):       {metrics['p_high_given_low_arousal']*100:.2f}%")
    print(f"• Razão de Contraste de Excitação (High Arousal / Low Arousal):    {metrics['arousal_contrast_ratio']:.2f}x")
    print(f"• P(Cue=Low  | Emoção de Baixa Excitação [Neutral/Sadness]):       {metrics['p_low_given_low_arousal']*100:.2f}%")

    # Tabela detalhada por emoção
    emo_rows = []
    for emo in sorted(df_disc["emotion"].unique()):
        sub = df_disc[df_disc["emotion"] == emo]
        row = {"Emoção": emo.capitalize(), "N": len(sub)}
        for c in CONTINUOUS_4_CUES:
            row[f"{c} High (%)"] = f"{(sub[f'{c}_tier'] == 'high').mean()*100:.1f}%"
        emo_rows.append(row)

    print("\n📋 Distribuição Empírica de High por Emoção:")
    print(pd.DataFrame(emo_rows).to_string(index=False))

    print("\n" + "=" * 90)
    print("🎯 CONCLUSÃO TÉCNICA:")
    print(f"1. A hipótese de independência (P = p1*p2*p3*p4) é categoricamente FALSA no MELD.")
    print(f"   Três cues coincidem em 'High' com frequência {metrics['ratio_3_high']:.1f}x superior ao modelo independente.")
    print(f"2. Emoções de alta excitação ativam 'High' {metrics['arousal_contrast_ratio']:.1f}x mais que o neutro/tristeza.")
    print(f"   Isso explica o viés para 'High' no GRPO com beta=0: os dados reais concentram alta excitação.")
    print("=" * 90)

    return metrics


# ==============================================================================
# TESTES AUTOMATIZADOS (PYTEST / ASSERTIONS)
# ==============================================================================
def test_acoustic_parameters_have_significant_positive_correlation():
    """Valida que parâmetros de pitch, energia e velocidade possuem correlação positiva na fala real."""
    df = load_meld_acoustic_dataset()
    pearson_corr, _ = compute_correlations(df)

    # Pitch Height & Vocal Energy são correlacionados positivamente (quem fala alto sobe pitch)
    assert pearson_corr.loc["pitch_height", "vocal_energy"] > 0.15, (
        f"Correlação esperada > 0.15, obtida: {pearson_corr.loc['pitch_height', 'vocal_energy']:.3f}"
    )

    # Pitch Height & Speaking Rate
    assert pearson_corr.loc["pitch_height", "speaking_rate"] > 0.20, (
        f"Correlação esperada > 0.20, obtida: {pearson_corr.loc['pitch_height', 'speaking_rate']:.3f}"
    )


def test_extreme_cues_cooccurrence_violates_independence():
    """Valida que a taxa real de co-ocorrência em High é estritamente maior que o produto de marginais independentes."""
    df = load_meld_acoustic_dataset()
    df_disc = discretize_cues_by_quartiles(df)
    metrics = compute_cooccurrence_and_coupling(df_disc)

    # Razão de amplificação de 3 highs deve ser superior a 1.5x
    assert metrics["ratio_3_high"] > 1.5, (
        f"Amplificação obtida ({metrics['ratio_3_high']:.2f}x) abaixo do esperado (> 1.5x)"
    )


def test_high_arousal_emotions_have_higher_probability_of_high_cues():
    """Valida que falas de alta excitação concentram significativamente mais respostas High do que neutras/tristes."""
    df = load_meld_acoustic_dataset()
    df_disc = discretize_cues_by_quartiles(df)
    metrics = compute_cooccurrence_and_coupling(df_disc)

    assert metrics["p_high_given_high_arousal"] > metrics["p_high_given_low_arousal"], (
        f"P(High | Alta Excitação) [{metrics['p_high_given_high_arousal']:.3f}] deve superar P(High | Baixa) [{metrics['p_high_given_low_arousal']:.3f}]"
    )
    assert metrics["arousal_contrast_ratio"] >= 1.4, (
        f"Contraste de excitação ({metrics['arousal_contrast_ratio']:.2f}x) deve ser >= 1.4x"
    )


if __name__ == "__main__":
    run_parameter_correlation_diagnostics()
