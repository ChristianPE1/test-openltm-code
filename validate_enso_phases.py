"""
FASE 2: Validación ENSO-aware para Tesis
=========================================

Este script separa el test set por fases ENSO y calcula métricas independientes.

FASES ENSO (basado en ONI index histórico):
- El Niño: 2015-2016, 2018-2019, 2023-2024
- La Niña: 2020-2021, 2021-2022, 2022-2023  
- Neutral: 2017, 2019-2020, 2024

HIPÓTESIS DE TESIS:
El modelo Timer-XL debe:
1. Mantener F1 > 0.75 en TODAS las fases ENSO
2. Mejor rendimiento en fases extremas (El Niño/La Niña) vs Neutral
3. Consistencia: |F1_ElNiño - F1_LaNiña| < 0.15
"""

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import os


# Definición de fases ENSO (basado en ONI index)
ENSO_PHASES = {
    "El_Niño": [
        ("2015-01-01", "2016-06-30"),  # El Niño fuerte 2015-16
        ("2018-10-01", "2019-06-30"),  # El Niño débil 2018-19
        ("2023-06-01", "2024-05-31"),  # El Niño 2023-24
    ],
    "La_Niña": [
        ("2020-08-01", "2021-05-31"),  # La Niña 2020-21
        ("2021-09-01", "2022-03-31"),  # La Niña 2021-22
        ("2022-09-01", "2023-02-28"),  # La Niña 2022-23
    ],
    "Neutral": [
        ("2017-01-01", "2017-12-31"),  # Neutral 2017
        ("2019-07-01", "2020-07-31"),  # Neutral 2019-20
        ("2024-06-01", "2024-12-31"),  # Neutral 2024
    ]
}


def assign_enso_phase(timestamp):
    """Asigna fase ENSO a un timestamp."""
    ts = pd.to_datetime(timestamp)
    
    for phase, periods in ENSO_PHASES.items():
        for start, end in periods:
            if pd.to_datetime(start) <= ts <= pd.to_datetime(end):
                return phase
    
    return "Undefined"  # Períodos de transición


def load_predictions_and_labels(checkpoint_dir, data_path):
    """
    Carga predicciones del modelo y labels verdaderos.
    
    Args:
        checkpoint_dir: Directorio con el checkpoint .pth
        data_path: Ruta al CSV con los datos y timestamps
    
    Returns:
        df: DataFrame con [timestamp, true_label, pred_label, pred_proba, enso_phase]
    """
    # Cargar datos originales con timestamps
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    
    # Asignar fase ENSO
    df['enso_phase'] = df['timestamp'].apply(assign_enso_phase)
    
    # Aquí deberías cargar las predicciones del modelo
    # Por ahora, placeholder - debes adaptarlo a tu pipeline
    print("⚠️ NOTA: Debes integrar carga de predicciones del checkpoint")
    print(f"   Checkpoint: {checkpoint_dir}")
    
    # Placeholder: asume que tienes un archivo de predicciones
    # pred_file = os.path.join(checkpoint_dir, 'predictions_test.npy')
    # preds = np.load(pred_file)
    # df['pred_label'] = preds['labels']
    # df['pred_proba_rain'] = preds['probas'][:, 1]
    
    return df


def calculate_metrics_by_phase(df):
    """
    Calcula métricas por fase ENSO.
    
    Returns:
        results: Dict con métricas por fase
    """
    results = {}
    
    for phase in ["El_Niño", "La_Niña", "Neutral", "Undefined"]:
        df_phase = df[df['enso_phase'] == phase]
        
        if len(df_phase) == 0:
            continue
        
        y_true = df_phase['rain_24h']
        y_pred = df_phase['pred_label']
        y_proba = df_phase['pred_proba_rain']
        
        results[phase] = {
            'n_samples': len(df_phase),
            'accuracy': (y_true == y_pred).mean(),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'f1_no_rain': f1_score(y_true, y_pred, pos_label=0),
            'f1_rain': f1_score(y_true, y_pred, pos_label=1),
            'recall_no_rain': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'recall_rain': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else None,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    return results


def plot_enso_comparison(results, output_dir):
    """Genera visualizaciones comparativas por fase ENSO."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Gráfico de barras: F1-Score por fase
    phases = [p for p in ["El_Niño", "La_Niña", "Neutral"] if p in results]
    f1_scores = [results[p]['f1'] for p in phases]
    f1_no_rain = [results[p]['f1_no_rain'] for p in phases]
    f1_rain = [results[p]['f1_rain'] for p in phases]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(phases))
    width = 0.25
    
    ax.bar(x - width, f1_scores, width, label='F1 Weighted', color='blue', alpha=0.7)
    ax.bar(x, f1_no_rain, width, label='F1 No Rain', color='orange', alpha=0.7)
    ax.bar(x + width, f1_rain, width, label='F1 Rain', color='green', alpha=0.7)
    
    ax.set_xlabel('Fase ENSO', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Rendimiento del Modelo por Fase ENSO', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.75, color='red', linestyle='--', label='Meta F1 > 0.75')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enso_f1_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Confusion Matrices por fase
    fig, axes = plt.subplots(1, len(phases), figsize=(15, 4))
    
    for idx, phase in enumerate(phases):
        cm = results[phase]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['No Rain', 'Rain'],
                    yticklabels=['No Rain', 'Rain'])
        axes[idx].set_title(f'{phase}\n(n={results[phase]["n_samples"]})')
        axes[idx].set_xlabel('Predicho')
        axes[idx].set_ylabel('Real')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enso_confusion_matrices.png'), dpi=300)
    plt.close()
    
    print(f"✅ Visualizaciones guardadas en: {output_dir}")


def generate_report(results, output_file):
    """Genera reporte en texto con métricas ENSO."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("📊 VALIDACIÓN ENSO-AWARE - RESULTADOS POR FASE\n")
        f.write("="*80 + "\n\n")
        
        for phase in ["El_Niño", "La_Niña", "Neutral", "Undefined"]:
            if phase not in results:
                continue
            
            r = results[phase]
            
            f.write(f"{'='*80}\n")
            f.write(f"🌊 FASE: {phase}\n")
            f.write(f"{'='*80}\n")
            f.write(f"   Samples: {r['n_samples']}\n")
            f.write(f"   Accuracy: {r['accuracy']:.4f}\n")
            f.write(f"   F1-Score (Weighted): {r['f1']:.4f}\n")
            f.write(f"   F1 No Rain: {r['f1_no_rain']:.4f}\n")
            f.write(f"   F1 Rain: {r['f1_rain']:.4f}\n")
            f.write(f"   Recall No Rain: {r['recall_no_rain']:.4f}\n")
            f.write(f"   Recall Rain: {r['recall_rain']:.4f}\n")
            
            if r['auc_roc'] is not None:
                f.write(f"   AUC-ROC: {r['auc_roc']:.4f}\n")
            
            f.write(f"\n   Confusion Matrix:\n")
            cm = r['confusion_matrix']
            f.write(f"                  Predicted\n")
            f.write(f"              No Rain  |  Rain\n")
            f.write(f"   Actual No Rain {cm[0,0]:5d}   | {cm[0,1]:5d}\n")
            f.write(f"          Rain    {cm[1,0]:5d}   | {cm[1,1]:5d}\n\n")
        
        # Análisis de consistencia
        phases = ["El_Niño", "La_Niña", "Neutral"]
        if all(p in results for p in phases):
            f.write("="*80 + "\n")
            f.write("📈 ANÁLISIS DE CONSISTENCIA\n")
            f.write("="*80 + "\n")
            
            f1_values = [results[p]['f1'] for p in phases]
            f1_range = max(f1_values) - min(f1_values)
            
            f.write(f"   F1-Score Range: {f1_range:.4f}\n")
            f.write(f"   Consistencia: {'✅ EXCELENTE' if f1_range < 0.10 else '✅ BUENA' if f1_range < 0.15 else '⚠️ MODERADA'}\n")
            f.write(f"   Meta: < 0.15 para consistencia aceptable\n\n")
            
            # Comparación El Niño vs La Niña
            diff_el_la = abs(results["El_Niño"]['f1'] - results["La_Niña"]['f1'])
            f.write(f"   |F1_ElNiño - F1_LaNiña|: {diff_el_la:.4f}\n")
            f.write(f"   Hipótesis cumplida: {'✅ SÍ' if diff_el_la < 0.15 else '❌ NO'}\n\n")
    
    print(f"✅ Reporte guardado en: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Validación ENSO-aware')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directorio con el checkpoint .pth')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Ruta al CSV con datos y timestamps')
    parser.add_argument('--output_dir', type=str, default='results/enso_validation',
                        help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🌊 VALIDACIÓN ENSO-AWARE")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}\n")
    
    # Cargar datos y predicciones
    df = load_predictions_and_labels(args.checkpoint_dir, args.data_path)
    
    # Calcular métricas por fase
    results = calculate_metrics_by_phase(df)
    
    # Generar visualizaciones
    plot_enso_comparison(results, args.output_dir)
    
    # Generar reporte
    report_file = os.path.join(args.output_dir, 'enso_validation_report.txt')
    generate_report(results, report_file)
    
    print("\n✅ Validación ENSO-aware completada")


if __name__ == '__main__':
    main()
