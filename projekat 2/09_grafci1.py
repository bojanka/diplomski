import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv(r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\rezultati\rezultati_svi.csv")
results_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\rezultati"

def get_color(ai):
    if ai < 1.0: return '#4CAF50'
    elif ai < 1.5: return '#FFC107'
    elif ai < 2.5: return '#FF9800'
    else: return '#F44336'

colors = [get_color(ai) for ai in df['ai_percent']]

# --- GRAFIK 1: Bar chart AI po svim 42 pacijentima ---
fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(df['case'], df['ai_percent'], color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(y=1.0, color='#4CAF50', linestyle='--', alpha=0.7, linewidth=1.5)
ax.axhline(y=1.5, color='#FFC107', linestyle='--', alpha=0.7, linewidth=1.5)
ax.axhline(y=2.5, color='#F44336', linestyle='--', alpha=0.7, linewidth=1.5)
ax.set_xlabel('Pacijent', fontsize=12)
ax.set_ylabel('Indeks asimetrije (%)', fontsize=12)
ax.set_title('Indeks asimetrije (AI) za svih 42 pacijenta — HaN-Seg dataset', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(df['case'])))
ax.set_xticklabels(df['case'], rotation=45, ha='right', fontsize=7)
ax.set_ylim(0, 4.0)
ax.grid(axis='y', alpha=0.3)
p1 = mpatches.Patch(color='#4CAF50', label='Prirodna varijacija (AI < 1.0%)')
p2 = mpatches.Patch(color='#FFC107', label='Blaga asimetrija (1.0-1.5%)')
p3 = mpatches.Patch(color='#FF9800', label='Umerena asimetrija (1.5-2.5%)')
p4 = mpatches.Patch(color='#F44336', label='Znacajna asimetrija (AI > 2.5%)')
ax.legend(handles=[p1,p2,p3,p4], fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(f'{results_path}\\stat1_bar_ai.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 1 sacuvan')

# --- GRAFIK 2: Histogram distribucije odstupanja ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['mean_mm'], bins=12, color='#2196F3', edgecolor='white', alpha=0.8)
axes[0].axvline(x=df['mean_mm'].mean(), color='#F44336', linestyle='--', linewidth=2,
    label=f'Prosek: {df["mean_mm"].mean():.3f} mm')
axes[0].axvline(x=2.0, color='#FF9800', linestyle='--', linewidth=1.5, label='Klinicki prag: 2.0 mm')
axes[0].set_xlabel('Prosecno odstupanje (mm)', fontsize=11)
axes[0].set_ylabel('Broj pacijenata', fontsize=11)
axes[0].set_title('Distribucija prosecnog odstupanja', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].hist(df['max_mm'], bins=12, color='#9C27B0', edgecolor='white', alpha=0.8)
axes[1].axvline(x=df['max_mm'].mean(), color='#F44336', linestyle='--', linewidth=2,
    label=f'Prosek: {df["max_mm"].mean():.3f} mm')
axes[1].axvline(x=8.8, color='#F44336', linestyle='-', linewidth=1.5, label='Hirurski prag: 8.8 mm')
axes[1].set_xlabel('Maksimalno odstupanje (mm)', fontsize=11)
axes[1].set_ylabel('Broj pacijenata', fontsize=11)
axes[1].set_title('Distribucija maksimalnog odstupanja', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_path}\\stat2_histogram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 2 sacuvan')

# --- GRAFIK 3: Scatter AI vs max odstupanje ---
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(df['ai_percent'], df['max_mm'], c=df['mean_mm'],
    cmap='RdYlGn_r', s=80, edgecolors='gray', linewidth=0.5, alpha=0.85)
for i, row in df.iterrows():
    if row['ai_percent'] > 2.5 or row['max_mm'] > 10:
        ax.annotate(row['case'], (row['ai_percent'], row['max_mm']),
            textcoords='offset points', xytext=(5, 5), fontsize=7)
ax.axhline(y=8.8, color='#F44336', linestyle='--', linewidth=1.5, label='Hirurski prag (8.8 mm)')
ax.axvline(x=2.5, color='#FF9800', linestyle='--', linewidth=1.5, label='Znacajna asimetrija (2.5%)')
plt.colorbar(sc, label='Prosecno odstupanje (mm)')
ax.set_xlabel('Indeks asimetrije (%)', fontsize=12)
ax.set_ylabel('Maksimalno odstupanje (mm)', fontsize=12)
ax.set_title('Scatter plot — AI vs maksimalno odstupanje', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_path}\\stat3_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 3 sacuvan')

# --- GRAFIK 4: Box plot po anatomskim regionima ---
fig, ax = plt.subplots(figsize=(10, 6))
region_data = [df['kondil_levo_mean'], df['kondil_desno_mean'],
               df['telo_levo_mean'], df['telo_desno_mean']]
region_labels = ['Kondil/Ramus\nLEVO', 'Kondil/Ramus\nDESNO', 'Telo\nLEVO', 'Telo\nDESNO']
bp = ax.boxplot(region_data, tick_labels=region_labels, patch_artist=True)
box_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.axhline(y=df[['kondil_levo_mean','kondil_desno_mean','telo_levo_mean','telo_desno_mean']].values.mean(),
    color='red', linestyle='--', linewidth=1.5, alpha=0.7,
    label=f'Ukupni prosek: {df["mean_mm"].mean():.3f} mm')
ax.set_ylabel('Prosecno odstupanje (mm)', fontsize=12)
ax.set_title('Distribucija asimetrije po anatomskim regionima\n(svih 42 pacijenta)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_path}\\stat4_boxplot.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 4 sacuvan')

# --- GRAFIK 5: Bar chart po regionima sa std ---
fig, ax = plt.subplots(figsize=(10, 5))
regioni = ['Kondil/Ramus Levo', 'Kondil/Ramus Desno', 'Telo Levo', 'Telo Desno']
proseci = [df['kondil_levo_mean'].mean(), df['kondil_desno_mean'].mean(),
           df['telo_levo_mean'].mean(), df['telo_desno_mean'].mean()]
std_devs = [df['kondil_levo_mean'].std(), df['kondil_desno_mean'].std(),
            df['telo_levo_mean'].std(), df['telo_desno_mean'].std()]
bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
bars = ax.bar(regioni, proseci, yerr=std_devs, color=bar_colors,
    alpha=0.8, edgecolor='white', capsize=6, linewidth=1.5)
for bar, val in zip(bars, proseci):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
        f'{val:.3f} mm', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Prosecno odstupanje (mm)', fontsize=12)
ax.set_title('Prosecno odstupanje po anatomskim regionima mandibule\n(sa standardnom devijacijom)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_path}\\stat5_regioni.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 5 sacuvan')

print('\nSvi grafici sacuvani!')