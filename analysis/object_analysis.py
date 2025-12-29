import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from statsmodels.stats.anova import AnovaRM # RM-ANOVA専用のモジュール
from scipy import stats

# ============================================================================
# 設定: ファイルパスと日本語フォント
# ============================================================================
# 分析対象のCSVファイル一覧
DATA_FILES = {
    '得点': {
        'path': '結果（客観指標） - 得点.csv',
        'label': '得点',
        'unit': '点'
    },
    '学習時間': {
        'path': '結果（客観指標） - 学習時間.csv',
        'label': '学習時間',
        'unit': '秒'
    },
    '視線移動距離': {
        'path': '結果（客観指標） - 視線移動距離.csv',
        'label': '視線移動距離',
        'unit': 'px'
    },
    'フレーム単位増加速度': {
        'path': '結果（客観指標） - フレーム単位の増加速度.csv',
        'label': 'フレーム単位の増加速度',
        'unit': 'px/frame'
    }
}

INPUT_DIR = 'results'
OUTPUT_DIR = 'analysis_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 日本語表示の設定
rcParams['font.family'] = 'MS Gothic'
rcParams['axes.unicode_minus'] = False

# ============================================================================
# ステップ1: 実験条件とカウンターバランスの定義
# ============================================================================
# 実験デザインを辞書型で定義。条件名から「配置手法」と「時間制限」を自動判別します。
CONDITIONS = {
    'A': {'name': '動的・制限あり', 'type': 'saliency', 'time_limit': '時間制限あり'},
    'B': {'name': '動的・制限なし', 'type': 'saliency', 'time_limit': '時間制限なし'},
    'C': {'name': '固定・制限あり', 'type': 'fixed', 'time_limit': '時間制限あり'},
    'D': {'name': '固定・制限なし', 'type': 'fixed', 'time_limit': '時間制限なし'},
}

# ラテン方格の定義（実験順序による慣れや疲労の影響を相殺するため）
LATIN_SQUARE_ORDERS = [
    ['A', 'B', 'C', 'D'],
    ['B', 'C', 'D', 'A'],
    ['C', 'D', 'A', 'B'],
    ['D', 'A', 'B', 'C']
]
PATTERN_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# ============================================================================
# ステップ2: データ分析関数の定義
# ============================================================================
def analyze_data(file_key, file_info):
    """
    指定されたCSVファイルに対して2元配置反復測定分散分析を実行
    
    Parameters:
    -----------
    file_key : str
        ファイルの識別キー（例: '得点'）
    file_info : dict
        ファイル情報（path, label, unit）
    """
    print("\n" + "="*80)
    print(f"【{file_info['label']}の分析】")
    print("="*80)
    
    # 出力ディレクトリの作成
    output_subdir = os.path.join(OUTPUT_DIR, file_key)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    
    # データの読み込み
    input_path = os.path.join(INPUT_DIR, file_info['path'])
    
    try:
        df = pd.read_csv(input_path, na_values=['NaN', 'Nan', 'nan', ''])
    except FileNotFoundError:
        print(f"⚠️ ファイルが見つかりません: {input_path}")
        print(f"スキップします。\n")
        return
    
    # 数値として扱う列を変換し、不完全なデータ（欠損値）を持つ被験者を削除
    numeric_cols = ['1', '2', '3', '4']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    if len(df) == 0:
        print(f"⚠️ 有効なデータがありません")
        print(f"スキップします。\n")
        return
    
    print(f"有効な被験者数: {len(df)}")
    
    # Wide形式からLong形式への変換
    long_data = []
    
    for index, row in df.iterrows():
        subject_id = row['実験番号']
        pattern_char = str(row['条件順序']).strip()
        
        if pattern_char in PATTERN_MAP:
            pattern_idx = PATTERN_MAP[pattern_char]
            latin_square = LATIN_SQUARE_ORDERS[pattern_idx]
            
            for block_num in range(1, 5):
                condition_key = latin_square[block_num - 1]
                score = row[str(block_num)]
                cond_info = CONDITIONS[condition_key]
                
                long_data.append({
                    '被験者ID': subject_id,
                    '配置手法': cond_info['type'],
                    '時間制限': cond_info['time_limit'],
                    'スコア': score
                })
    
    df_long = pd.DataFrame(long_data)
    
    # 正規性検定
    W, shapiro_p_value = stats.shapiro(df_long['スコア'])
    print(f'\nShapiro-Wilk検定: W={W:.4f}, p={shapiro_p_value:.4f}')
    if shapiro_p_value < 0.05:
        print("⚠️ データは正規分布に従っていません（p < 0.05）")
    else:
        print("✓ データは正規分布に従っています（p >= 0.05）")
    
    # 2元配置反復測定分散分析
    anova_result = AnovaRM(data=df_long, depvar='スコア', subject='被験者ID', 
                           within=['配置手法', '時間制限']).fit()
    
    print(f"\n{anova_result.summary()}")
    
    # p値の取得
    placement_p = anova_result.anova_table.loc['配置手法', 'Pr > F']
    time_p = anova_result.anova_table.loc['時間制限', 'Pr > F']
    interaction_p = anova_result.anova_table.loc['配置手法:時間制限', 'Pr > F']
    
    print(f"\n配置手法のp値: {placement_p:.4f} → {'✓ 有意' if placement_p < 0.05 else '× 有意でない'}")
    print(f"時間制限のp値: {time_p:.4f} → {'✓ 有意' if time_p < 0.05 else '× 有意でない'}")
    print(f"交互作用のp値: {interaction_p:.4f} → {'✓ 有意' if interaction_p < 0.05 else '× 有意でない'}")
    
    # ============================================================================
    # グラフの作成
    # ============================================================================
    
    # 記述統計の計算
    interaction_stats = df_long.groupby(['配置手法', '時間制限'])['スコア'].agg(['mean', 'std', 'count'])
    interaction_stats['se'] = interaction_stats['std'] / np.sqrt(interaction_stats['count'])
    
    placement_stats = df_long.groupby('配置手法')['スコア'].agg(['mean', 'std', 'count'])
    placement_stats['se'] = placement_stats['std'] / np.sqrt(placement_stats['count'])
    
    time_stats = df_long.groupby('時間制限')['スコア'].agg(['mean', 'std', 'count'])
    time_stats['se'] = time_stats['std'] / np.sqrt(time_stats['count'])
    
    # ============================================================================
    # 図1: 交互作用プロット
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for method, color, marker in [('fixed', '#e74c3c', 'o'), ('saliency', '#3498db', 's')]:
        subset = interaction_stats.xs(method, level='配置手法')
        x_labels = ['時間制限なし', '時間制限あり']
        means = [subset.loc['時間制限なし', 'mean'], subset.loc['時間制限あり', 'mean']]
        errs = [subset.loc['時間制限なし', 'se'], subset.loc['時間制限あり', 'se']]
        
        ax.errorbar(x_labels, means, yerr=errs, 
                    label=f'{method.upper()}配置', 
                    color=color, marker=marker, markersize=12, 
                    linewidth=3, capsize=8, capthick=2)

    ax.set_title(f'配置手法 × 時間制限 の交互作用\n(p = {interaction_p:.4f})', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('時間制限', fontsize=14, fontweight='bold')
    ax.set_ylabel('平均スコア', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 有意性マーカー
    if interaction_p < 0.05:
        significance = '**' if interaction_p < 0.01 else '*'
        ax.text(0.5, 0.95, f'有意な交互作用あり {significance}',
                transform=ax.transAxes, ha='center', fontsize=13,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'interaction_plot.png'), dpi=300, bbox_inches='tight')
    print(f"\n交互作用プロットを保存: {output_subdir}/interaction_plot.png")
    plt.close()

    # ============================================================================
    # 図2: 主効果（配置手法と時間制限）
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 配置手法の主効果
    ax1 = axes[0]
    methods = ['fixed', 'saliency']
    method_labels = ['FIXED\n(固定配置)', 'SALIENCY\n(動的配置)']
    means = [placement_stats.loc[m, 'mean'] for m in methods]
    errs = [placement_stats.loc[m, 'se'] for m in methods]
    colors = ['#e74c3c', '#3498db']

    bars = ax1.bar(method_labels, means, yerr=errs, capsize=10,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    ax1.set_ylabel('平均スコア', fontsize=13, fontweight='bold')
    ax1.set_title(f'配置手法の主効果\n(p = {placement_p:.4f})', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 平均値を表示
    for bar, mean, err in zip(bars, means, errs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + err + max(means)*0.02,
                 f'{mean:.1f}', ha='center', va='bottom', 
                 fontsize=12, fontweight='bold')

    # 有意性マーカー
    if placement_p < 0.05:
        y_max = max([m + e for m, e in zip(means, errs)])
        ax1.plot([0, 1], [y_max * 1.1, y_max * 1.1], 'k-', linewidth=2)
        significance = '**' if placement_p < 0.01 else '*'
        ax1.text(0.5, y_max * 1.12, significance, ha='center', 
                 fontsize=20, fontweight='bold')

    # 時間制限の主効果
    ax2 = axes[1]
    time_labels = ['時間制限なし', '時間制限あり']
    means2 = [time_stats.loc[t, 'mean'] for t in time_labels]
    errs2 = [time_stats.loc[t, 'se'] for t in time_labels]
    colors2 = ['#f39c12', '#2ecc71']

    bars2 = ax2.bar(time_labels, means2, yerr=errs2, capsize=10,
                    color=colors2, alpha=0.7, edgecolor='black', linewidth=2)

    ax2.set_ylabel('平均スコア', fontsize=13, fontweight='bold')
    ax2.set_title(f'時間制限の主効果\n(p = {time_p:.4f})', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 平均値を表示
    for bar, mean, err in zip(bars2, means2, errs2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + err + max(means2)*0.02,
                 f'{mean:.1f}', ha='center', va='bottom', 
                 fontsize=12, fontweight='bold')

    # 有意性マーカー
    if time_p < 0.05:
        y_max = max([m + e for m, e in zip(means2, errs2)])
        ax2.plot([0, 1], [y_max * 1.1, y_max * 1.1], 'k-', linewidth=2)
        significance = '**' if time_p < 0.01 else '*'
        ax2.text(0.5, y_max * 1.12, significance, ha='center', 
                 fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'main_effects.png'), dpi=300, bbox_inches='tight')
    print(f"主効果のグラフを保存: {output_subdir}/main_effects.png")
    plt.close()

    # ============================================================================
    # 図3: 4条件の比較
    # ============================================================================
    fig, ax = plt.subplots(figsize=(12, 7))

    conditions_order = [
        ('saliency', '時間制限あり'),
        ('saliency', '時間制限なし'),
        ('fixed', '時間制限あり'),
        ('fixed', '時間制限なし'),
    ]

    condition_labels = [
        'Saliency\n時間制限あり',
        'Saliency\n時間制限なし',
        'Fixed\n時間制限あり',
        'Fixed\n時間制限なし'
    ]

    means = [interaction_stats.loc[cond, 'mean'] for cond in conditions_order]
    errs = [interaction_stats.loc[cond, 'se'] for cond in conditions_order]
    colors_4 = ['#e74c3c', '#c0392b', '#3498db', '#2980b9']

    x_pos = np.arange(len(condition_labels))
    bars = ax.bar(x_pos, means, yerr=errs, capsize=10,
                  color=colors_4, alpha=0.6, edgecolor='black', linewidth=2)

    ax.set_ylabel('スコア', fontsize=13, fontweight='bold')
    ax.set_title('4条件の比較', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(condition_labels, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

# 平均値を表示
    for i, (bar, mean, err) in enumerate(zip(bars, means, errs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + max(means)*0.02,
                f'{mean:.1f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'all_conditions.png'), dpi=300, bbox_inches='tight')
    print(f"4条件比較グラフを保存: {output_subdir}/all_conditions.png")
    plt.close()
    
    print(f"\n✓ {file_info['label']}の分析完了！")
    print(f"  結果保存先: {output_subdir}/")


# ============================================================================
# メイン処理: 全CSVファイルに対してループ実行
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("2元配置反復測定分散分析 - 一括処理")
    print("="*80)
    
    for file_key, file_info in DATA_FILES.items():
        analyze_data(file_key, file_info)
    
    print("\n" + "="*80)
    print("すべてのファイルの分析が完了しました！")
    print("="*80)
    print(f"\n結果は {OUTPUT_DIR}/ 以下の各サブディレクトリに保存されています。")
