"""
FASE 3: Análisis Regional para Tesis
====================================

Este script analiza el rendimiento del modelo por regiones de Perú.

REGIONES (basado en influencia ENSO):
- Costa Norte (4°S - 8°S): Máxima influencia ENSO (Piura, Tumbes, Lambayeque)
- Costa Centro (8°S - 14°S): Influencia moderada (Lima, Callao, Ica)
- Costa Sur (14°S - 18°S): Mínima influencia ENSO (Arequipa, Moquegua, Tacna)

HIPÓTESIS DE TESIS:
1. Costa Norte debe tener MEJOR F1 (mayor señal ENSO → más predecible)
2. Costa Sur debe tener PEOR F1 (menor influencia ENSO)
3. Gradiente Norte→Sur: F1_Norte > F1_Centro > F1_Sur
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


# Definición de regiones (latitudes)
REGIONS = {
    "Costa_Norte": (-8.0, -4.0),    # Máxima influencia ENSO
    "Costa_Centro": (-14.0, -8.0),  # Influencia moderada
    "Costa_Sur": (-18.0, -14.0),    # Mínima influencia ENSO
}


def assign_region(lat):
    """Asigna región geográfica basada en latitud."""
    for region, (lat_min, lat_max) in REGIONS.items():
        if lat_min <= lat <= lat_max:
            return region
    return "Fuera_de_rango"


def calculate_regional_metrics(df):
    """
    Calcula métricas por región.
    
    Args:
        df: DataFrame con columnas [region, rain_24h, pred_label, pred_proba_rain]
    
    Returns:
        results: Dict con métricas por región
    """
    results = {}
    
    for region in ["Costa_Norte", "Costa_Centro", "Costa_Sur"]:
        df_region = df[df['region'] == region]
        
        if len(df_region) == 0:
            print(f"⚠️ No hay datos para región: {region}")
            continue
        
        y_true = df_region['rain_24h']
        y_pred = df_region['pred_label']
        y_proba = df_region['pred_proba_rain']
        
        results[region] = {
            'n_samples': len(df_region),
            'rain_prevalence': y_true.mean(),  # Proporción de lluvias
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


def plot_regional_comparison(results, output_dir):
    """Genera visualizaciones regionales."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    regions = ["Costa_Norte", "Costa_Centro", "Costa_Sur"]
    regions = [r for r in regions if r in results]
    
    # 1. Gráfico de barras: F1-Score por región
    f1_scores = [results[r]['f1'] for r in regions]
    f1_no_rain = [results[r]['f1_no_rain'] for r in regions]
    f1_rain = [results[r]['f1_rain'] for r in regions]
    rain_prev = [results[r]['rain_prevalence'] for r in regions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # F1-Scores
    x = np.arange(len(regions))
    width = 0.25
    
    ax1.bar(x - width, f1_scores, width, label='F1 Weighted', color='blue', alpha=0.7)
    ax1.bar(x, f1_no_rain, width, label='F1 No Rain', color='orange', alpha=0.7)
    ax1.bar(x + width, f1_rain, width, label='F1 Rain', color='green', alpha=0.7)
    
    ax1.set_xlabel('Región', fontsize=12)
    ax1.set_ylabel('F1-Score', fontsize=12)
    ax1.set_title('Rendimiento del Modelo por Región', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([r.replace('_', ' ') for r in regions], rotation=15)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.75, color='red', linestyle='--', label='Meta F1 > 0.75')
    
    # Prevalencia de lluvia
    ax2.bar(regions, rain_prev, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Región', fontsize=12)
    ax2.set_ylabel('Proporción de Lluvias', fontsize=12)
    ax2.set_title('Prevalencia de Lluvia por Región', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([r.replace('_', ' ') for r in regions], rotation=15)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regional_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Confusion Matrices por región
    fig, axes = plt.subplots(1, len(regions), figsize=(15, 4))
    
    for idx, region in enumerate(regions):
        cm = results[region]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                    xticklabels=['No Rain', 'Rain'],
                    yticklabels=['No Rain', 'Rain'])
        axes[idx].set_title(f'{region.replace("_", " ")}\n(n={results[region]["n_samples"]})')
        axes[idx].set_xlabel('Predicho')
        axes[idx].set_ylabel('Real')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regional_confusion_matrices.png'), dpi=300)
    plt.close()
    
    print(f"✅ Visualizaciones guardadas en: {output_dir}")


def generate_regional_report(results, output_file):
    """Genera reporte regional."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("🗺️ ANÁLISIS REGIONAL - RENDIMIENTO POR ZONA\n")
        f.write("="*80 + "\n\n")
        
        for region in ["Costa_Norte", "Costa_Centro", "Costa_Sur"]:
            if region not in results:
                continue
            
            r = results[region]
            
            f.write(f"{'='*80}\n")
            f.write(f"📍 REGIÓN: {region.replace('_', ' ')}\n")
            f.write(f"{'='*80}\n")
            f.write(f"   Samples: {r['n_samples']}\n")
            f.write(f"   Rain Prevalence: {r['rain_prevalence']:.2%}\n")
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
        
        # Análisis de gradiente Norte→Sur
        if all(r in results for r in ["Costa_Norte", "Costa_Centro", "Costa_Sur"]):
            f.write("="*80 + "\n")
            f.write("📊 ANÁLISIS DE GRADIENTE ENSO (Norte → Sur)\n")
            f.write("="*80 + "\n")
            
            f1_norte = results["Costa_Norte"]['f1']
            f1_centro = results["Costa_Centro"]['f1']
            f1_sur = results["Costa_Sur"]['f1']
            
            f.write(f"   F1 Norte:  {f1_norte:.4f}\n")
            f.write(f"   F1 Centro: {f1_centro:.4f}\n")
            f.write(f"   F1 Sur:    {f1_sur:.4f}\n\n")
            
            gradiente_correcto = f1_norte >= f1_centro >= f1_sur
            
            f.write(f"   Hipótesis (Norte > Centro > Sur): ")
            f.write(f"{'✅ CUMPLIDA' if gradiente_correcto else '❌ NO CUMPLIDA'}\n")
            
            if gradiente_correcto:
                f.write(f"\n   ✅ El modelo captura correctamente la influencia ENSO:\n")
                f.write(f"      Mayor F1 en Costa Norte (máxima influencia ENSO)\n")
                f.write(f"      Menor F1 en Costa Sur (mínima influencia ENSO)\n")
            else:
                f.write(f"\n   ⚠️ El modelo NO captura el gradiente ENSO esperado\n")
                f.write(f"      Posibles causas:\n")
                f.write(f"      - Datos insuficientes en alguna región\n")
                f.write(f"      - Otros factores climáticos dominantes\n")
                f.write(f"      - Necesidad de features ENSO-aware explícitas\n")
    
    print(f"✅ Reporte guardado en: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Análisis Regional')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Ruta al CSV con datos, predicciones y coordenadas')
    parser.add_argument('--output_dir', type=str, default='results/regional_analysis',
                        help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🗺️ ANÁLISIS REGIONAL")
    print("="*80)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}\n")
    
    # Cargar datos
    df = pd.read_csv(args.data_path)
    
    # Asignar región si no existe
    if 'region' not in df.columns:
        print("⚠️ Columna 'region' no encontrada, asignando por latitud...")
        if 'latitude' in df.columns:
            df['region'] = df['latitude'].apply(assign_region)
        else:
            raise ValueError("Se requiere columna 'latitude' o 'region' en el CSV")
    
    # Calcular métricas
    results = calculate_regional_metrics(df)
    
    # Visualizaciones
    plot_regional_comparison(results, args.output_dir)
    
    # Reporte
    report_file = os.path.join(args.output_dir, 'regional_analysis_report.txt')
    generate_regional_report(results, report_file)
    
    print("\n✅ Análisis regional completado")


if __name__ == '__main__':
    main()
