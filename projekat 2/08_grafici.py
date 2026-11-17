import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Putanje do CSV fajlova - prilagodi svojoj lokaciji
df = pd.read_csv(r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\rezultati\parametri_pouzdani.csv")
df2 = pd.read_csv(r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\rezultati\rezultati_svi.csv")
results_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\rezultati"

# Boje po AI kategorijama
def get_color(ai):
    if ai < 1.0: return '#4CAF50'
    elif ai < 1.5: return '#FFC107'
    elif ai < 2.5: return '#FF9800'
    else: return '#F44336'

colors_ai = [get_color(ai) for ai in df2['ai_percent']]

# --- GRAFIK 1: Površina ---
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(df['case'], df['povrsina_mm2'], color=colors_ai, edgecolor='white', linewidth=0.5)
ax.axhline(y=df['povrsina_mm2'].mean(), color='black', linestyle='--', linewidth=1.5,
    label=f'Prosek: {df["povrsina_mm2"].mean():.0f} mm²')
ax.set_xlabel('Pacijent', fontsize=11)
ax.set_ylabel('Površina (mm²)', fontsize=11)
ax.set_title('Ukupna površina mandibule po pacijentu', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['case'], rotation=45, ha='right', fontsize=7)
p1 = mpatches.Patch(color='#4CAF50', label='AI < 1.0%')
p2 = mpatches.Patch(color='#FFC107', label='AI 1.0-1.5%')
p3 = mpatches.Patch(color='#FF9800', label='AI 1.5-2.5%')
p4 = mpatches.Patch(color='#F44336', label='AI > 2.5%')
ax.legend(handles=[p1,p2,p3,p4], fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_path}\\g1_povrsina.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 1 sacuvan')

# --- GRAFIK 2: Hausdorff i ICP ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(df['case'], df['hausdorff_mm'], color=colors_ai, edgecolor='white', linewidth=0.5)
axes[0].axhline(y=df['hausdorff_mm'].mean(), color='black', linestyle='--', linewidth=1.5,
    label=f'Prosek: {df["hausdorff_mm"].mean():.2f} mm')
axes[0].axhline(y=8.8, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Hirurski prag: 8.8mm')
axes[0].set_title('Hausdorff rastojanje (mm)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Pacijent', fontsize=10)
axes[0].set_ylabel('mm', fontsize=10)
axes[0].set_xticks(range(len(df)))
axes[0].set_xticklabels(df['case'], rotation=90, fontsize=6)
axes[0].legend(fontsize=8)
axes[0].grid(axis='y', alpha=0.3)
axes[1].bar(df['case'], df['icp_mm'], color=colors_ai, edgecolor='white', linewidth=0.5)
axes[1].axhline(y=df['icp_mm'].mean(), color='black', linestyle='--', linewidth=1.5,
    label=f'Prosek: {df["icp_mm"].mean():.3f} mm')
axes[1].set_title('ICP odstupanje (mm)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Pacijent', fontsize=10)
axes[1].set_ylabel('mm', fontsize=10)
axes[1].set_xticks(range(len(df)))
axes[1].set_xticklabels(df['case'], rotation=90, fontsize=6)
axes[1].legend(fontsize=8)
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_path}\\g2_hausdorff_icp.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 2 sacuvan')

# --- GRAFIK 3: Zakrivljenost ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(df['case'], df['zakrivljenost_mean'], color=colors_ai, edgecolor='white', linewidth=0.5)
axes[0].axhline(y=df['zakrivljenost_mean'].mean(), color='black', linestyle='--', linewidth=1.5,
    label=f'Prosek: {df["zakrivljenost_mean"].mean():.4f}')
axes[0].set_title('Srednja zakrivljenost površine mandibule', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Pacijent', fontsize=10)
axes[0].set_ylabel('Zakrivljenost (mm⁻¹)', fontsize=10)
axes[0].set_xticks(range(len(df)))
axes[0].set_xticklabels(df['case'], rotation=90, fontsize=6)
axes[0].legend(fontsize=8)
axes[0].grid(axis='y', alpha=0.3)
sc = axes[1].scatter(df['zakrivljenost_mean'], df2['ai_percent'],
    c=df2['mean_mm'], cmap='RdYlGn_r', s=60, edgecolors='gray', linewidth=0.5)
plt.colorbar(sc, ax=axes[1], label='Prosečno odstupanje (mm)')
axes[1].set_xlabel('Srednja zakrivljenost (mm⁻¹)', fontsize=10)
axes[1].set_ylabel('AI indeks (%)', fontsize=10)
axes[1].set_title('Zakrivljenost vs AI indeks', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_path}\\g3_zakrivljenost.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 3 sacuvan')

# --- GRAFIK 4: PCA ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im = axes[0].scatter(df['pca_var1'], df['pca_var2'], c=df2['ai_percent'],
    cmap='RdYlGn_r', s=70, edgecolors='gray', linewidth=0.5)
plt.colorbar(im, ax=axes[0], label='AI (%)')
for i, row in df.iterrows():
    if df2.loc[i, 'ai_percent'] > 2.5:
        axes[0].annotate(row['case'], (row['pca_var1'], row['pca_var2']),
            textcoords='offset points', xytext=(5,5), fontsize=7)
axes[0].set_xlabel('PCA var1 — glavna osa', fontsize=10)
axes[0].set_ylabel('PCA var2 — poprečna osa', fontsize=10)
axes[0].set_title('PCA oblika mandibule\n(boja = AI indeks)', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)
pca_means = [df['pca_var1'].mean(), df['pca_var2'].mean(), df['pca_var3'].mean()]
bar_cols = ['#2196F3', '#4CAF50', '#FF9800']
bars = axes[1].bar(['Osa 1\n(glavna)', 'Osa 2\n(poprečna)', 'Osa 3\n(vertikalna)'],
    [v*100 for v in pca_means], color=bar_cols, alpha=0.8, edgecolor='white')
for bar, val in zip(bars, pca_means):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
        f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Objašnjena varijansa (%)', fontsize=10)
axes[1].set_title('PCA — distribucija varijanse\npo osama oblika mandibule', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_path}\\g4_pca.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 4 sacuvan')

# --- GRAFIK 5: Korelacijska matrica ---
fig, ax = plt.subplots(figsize=(9, 7))
cols = ['hausdorff_mm', 'icp_mm', 'povrsina_mm2', 'zakrivljenost_mean', 'pca_var1', 'pca_var2']
labels = ['Hausdorff', 'ICP', 'Površina', 'Zakrivljenost', 'PCA var1', 'PCA var2']
corr = df[cols].corr()
im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Pearsonov koeficijent korelacije')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(labels, fontsize=10)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=9,
            color='black' if abs(corr.iloc[i,j]) < 0.7 else 'white')
ax.set_title('Korelacijska matrica morfoloških parametara', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{results_path}\\g5_korelacija.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 5 sacuvan')

# --- GRAFIK 6: Scatter Hausdorff vs AI ---
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df['hausdorff_mm'], df2['ai_percent'], c=df['povrsina_mm2'],
    cmap='Blues', s=80, edgecolors='gray', linewidth=0.5, alpha=0.9)
plt.colorbar(sc, label='Površina (mm²)')
for i, row in df.iterrows():
    if df2.loc[i,'ai_percent'] > 2.5 or row['hausdorff_mm'] > 11:
        ax.annotate(row['case'], (row['hausdorff_mm'], df2.loc[i,'ai_percent']),
            textcoords='offset points', xytext=(5,5), fontsize=8)
ax.axvline(x=8.8, color='red', linestyle='--', linewidth=1.5, label='Hirurski prag 8.8mm')
ax.axhline(y=2.5, color='orange', linestyle='--', linewidth=1.5, label='Značajna asimetrija 2.5%')
corr_val = np.corrcoef(df['hausdorff_mm'], df2['ai_percent'])[0,1]
ax.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=ax.transAxes, fontsize=12,
    fontweight='bold', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('Hausdorff rastojanje (mm)', fontsize=12)
ax.set_ylabel('AI indeks (%)', fontsize=12)
ax.set_title('Korelacija Hausdorff rastojanja i AI indeksa', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_path}\\g6_scatter_hausdorff_ai.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('Grafik 6 sacuvan')

print('\nSvi grafici sacuvani!')