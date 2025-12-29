import pandas as pd
import numpy as np
import os
from scipy import stats
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.family'] = 'MS Gothic'
rcParams['axes.unicode_minus'] = False

# ============================================================================
# Constants from furigana_web_app_time.py
# ============================================================================
CONDITIONS = {
    'A': {'name': '条件A (動的配置・時間固定あり)', 'type': 'saliency', 'time_limit': True},
    'B': {'name': '条件B (動的配置・時間固定なし)', 'type': 'saliency', 'time_limit': False},
    'C': {'name': '条件C (固定配置・時間固定あり)', 'type': 'fixed', 'time_limit': True},
    'D': {'name': '条件D (固定配置・時間固定なし)', 'type': 'fixed', 'time_limit': False},
}

# ラテン方格の順序パターン (Indices 0-3 correspond to Patterns A-D set in the experiment)
LATIN_SQUARE_ORDERS = [
    ['A', 'B', 'C', 'D'],
    ['B', 'C', 'D', 'A'],
    ['C', 'D', 'A', 'B'],
    ['D', 'A', 'B', 'C']
]

PATTERN_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

def analyze_questionnaire():
    input_path = os.path.join('results', '結果（主観指標） - アンケート（条件ごと）.csv')
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print(f"Reading {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 実験順序がブロック番号(1-4)であると仮定
    # 実験条件がパターンの開始条件(A-D)であると仮定

    # Store derived conditions
    real_conditions = []
    condition_types = []
    time_limits = []

    for index, row in df.iterrows():
        try:
            pattern_char = str(row['実験条件']).strip()
            # 実験順序が空またはNaNの場合の対策
            if pd.isna(row['実験順序']): # NaN だったら
                real_conditions.append(None)
                condition_types.append(None)
                time_limits.append(None)
                continue

            block_num = int(row['実験順序'])
            
            if pattern_char in PATTERN_MAP and 1 <= block_num <= 4:
                p_idx = PATTERN_MAP[pattern_char]
                # Determine the actual condition (A, B, C, or D) used in this block
                real_cond_char = LATIN_SQUARE_ORDERS[p_idx][block_num - 1]
                
                real_conditions.append(real_cond_char)
                
                # Metadata
                cond_info = CONDITIONS[real_cond_char]
                condition_types.append(cond_info['type'])
                time_limits.append(cond_info['time_limit'])
            else:
                print(f"Warning: Invalid pattern '{pattern_char}' or block '{block_num}' at row {index}")
                real_conditions.append(None)
                condition_types.append(None)
                time_limits.append(None)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            real_conditions.append(None)
            condition_types.append(None)
            time_limits.append(None)

    df['RealCondition'] = real_conditions
    df['ConditionType'] = condition_types
    df['TimeLimit'] = time_limits

    # Remove rows where condition could not be determined
    valid_df = df.dropna(subset=['RealCondition'])
    print(f"Valid responses: {len(valid_df)} / {len(df)}")

    # Identify question columns (Indices 4 to 20, assuming format matches 'アンケート条件.csv')
    # Column 0: Timestamp
    # Column 1: Exp ID
    # Column 2: Name
    # Column 3: Pattern
    # Column 4-20: Questions (17 columns)
    # Column 21: Comment
    # Column 22: Order
    
    # Get column names for questions
    question_columns = df.columns[4:21].tolist()
    
    print("\n--- 集計対象の質問項目 ---")
    for i, col in enumerate(question_columns):
        print(f"Q{i+1}: {col}")

    # =========================================================
    # スコア反転処理 (Reverse scoring for negative questions)
    # =========================================================
    # これらの質問は否定的な内容なので、スコアを反転させる (1→7, 2→6, 3→5, 4→4, 5→3, 6→2, 7→1)
    negative_questions = [
        'テキストの配置によって、集中が妨げられた。',
        'テキストを読むために余計な精神的努力が必要だと感じた。',
        'テキストのせいで学習のリズムが崩れた。',
        'テキストの位置が不快だった。'
    ]
    
    print("\n\n=== スコア反転処理 ===")
    print("以下の質問項目のスコアを反転します (1→7, 2→6, etc.):")
    for q in negative_questions:
        if q in question_columns:
            print(f"  - {q}")
            # 反転: 8 - x (1→7, 2→6, 3→5, 4→4, 5→3, 6→2, 7→1)
            valid_df[q] = 8 - valid_df[q]
        else:
            print(f"  Warning: Question '{q}' not found in columns")

    # =========================================================
    # 1. Group by Condition (A, B, C, D)
    # =========================================================
    print("\n\n=== 条件別平均値 (A, B, C, D) ===(反転後)")
    grouped_mean = valid_df.groupby('RealCondition')[question_columns].mean()
    grouped_std = valid_df.groupby('RealCondition')[question_columns].std()
    grouped_count = valid_df.groupby('RealCondition')['RealCondition'].count()

    print(grouped_mean)
    print(grouped_std)
    print("\nサンプル数:")
    print(grouped_count)

    # Save
    output_mean = os.path.join('results', 'summary_by_condition_mean.csv')
    grouped_mean.to_csv(output_mean, encoding='utf-8-sig')
    print(f"\nSaved mean summary to {output_mean}")

    # RM-ANOVA分析
    anova_results_list = []
    
    # グラフ保存用ディレクトリの作成
    graph_dir = os.path.join('results', 'graphs')
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    for i, q_col in enumerate(question_columns, 1):
        print(f"\n\n{'='*80}")
        print(f"【Q{i}: {q_col}】")
        print('='*80)
        
        long_data = [] # 質問ごとに初期化
        for _, row in valid_df.iterrows():
            cond_info = CONDITIONS.get(row['RealCondition'])
            if cond_info:
                long_data.append({
                    '被験者ID': row['実験番号'],
                    '配置手法': cond_info['type'],
                    '時間制限': 'あり' if cond_info['time_limit'] else 'なし',
                    'スコア': row[q_col]
                })
        
        df_long = pd.DataFrame(long_data)
        
        # 正規性検定
        W, shapiro_p = stats.shapiro(df_long['スコア'])
        print(f"\nShapiro-Wilk検定: W={W:.4f}, p={shapiro_p:.4f}")
        
        if shapiro_p < 0.05:
            print("⚠️ データは正規分布に従っていません（p < 0.05）")
            print("   → ウィルコクソンの符号順位検定を使用します")
            normality_status = "非正規"
            
            # ノンパラメトリック検定（ウィルコクソンの符号順位検定）
            try:
                # 配置手法の比較（時間制限を無視）
                # 各被験者について、saliencyとfixedのスコアを比較
                saliency_scores = df_long[df_long['配置手法'] == 'saliency'].groupby('被験者ID')['スコア'].mean()
                fixed_scores = df_long[df_long['配置手法'] == 'fixed'].groupby('被験者ID')['スコア'].mean()
                
                # 共通の被験者IDでソート
                common_ids = sorted(set(saliency_scores.index) & set(fixed_scores.index))
                saliency_vals = [saliency_scores[id] for id in common_ids]
                fixed_vals = [fixed_scores[id] for id in common_ids]
                
                wilcoxon_placement = stats.wilcoxon(saliency_vals, fixed_vals)
                placement_p = wilcoxon_placement.pvalue
                placement_stat = wilcoxon_placement.statistic
                
                print(f"\n【配置手法の比較（ウィルコクソン検定）】")
                print(f"  統計量: {placement_stat:.4f}")
                print(f"  p値: {placement_p:.4f} → {'✓ 有意' if placement_p < 0.05 else '× 有意でない'}")
                
                # 時間制限の比較（配置手法を無視）
                time_yes_scores = df_long[df_long['時間制限'] == 'あり'].groupby('被験者ID')['スコア'].mean()
                time_no_scores = df_long[df_long['時間制限'] == 'なし'].groupby('被験者ID')['スコア'].mean()
                
                common_ids_time = sorted(set(time_yes_scores.index) & set(time_no_scores.index))
                time_yes_vals = [time_yes_scores[id] for id in common_ids_time]
                time_no_vals = [time_no_scores[id] for id in common_ids_time]
                
                wilcoxon_time = stats.wilcoxon(time_yes_vals, time_no_vals)
                time_p = wilcoxon_time.pvalue
                time_stat = wilcoxon_time.statistic
                
                print(f"\n【時間制限の比較（ウィルコクソン検定）】")
                print(f"  統計量: {time_stat:.4f}")
                print(f"  p値: {time_p:.4f} → {'✓ 有意' if time_p < 0.05 else '× 有意でない'}")
                
                # 交互作用は検定できない
                interaction_p = np.nan
                print(f"\n【交互作用】")
                print(f"  ノンパラメトリック検定では交互作用は検定できません")
                
                # 記述統計の計算（グラフ用）
                interaction_stats = df_long.groupby(['配置手法', '時間制限'])['スコア'].agg(['mean', 'std', 'count'])
                interaction_stats['se'] = interaction_stats['std'] / np.sqrt(interaction_stats['count'])
                
                # グラフ作成
                fig, ax = plt.subplots(figsize=(10, 7))
                
                for method, color, marker in [('fixed', '#e74c3c', 'o'), ('saliency', '#3498db', 's')]:
                    subset = interaction_stats.xs(method, level='配置手法')
                    x_labels = ['時間制限なし', '時間制限あり']
                    means = [subset.loc['なし', 'mean'], subset.loc['あり', 'mean']]
                    errs = [subset.loc['なし', 'se'], subset.loc['あり', 'se']]
                    
                    ax.errorbar(x_labels, means, yerr=errs, 
                                label=f'{method.upper()}配置', 
                                color=color, marker=marker, markersize=12, 
                                linewidth=3, capsize=8, capthick=2)
                
                # タイトルと軸ラベル
                title = f'Q{i}: {q_col[:30]}...' if len(q_col) > 30 else f'Q{i}: {q_col}'
                ax.set_title(f'{title}\n(ウィルコクソン検定使用)', 
                             fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('時間制限', fontsize=13, fontweight='bold')
                ax.set_ylabel('平均スコア (1-7)', fontsize=13, fontweight='bold')
                ax.legend(fontsize=11, loc='best', framealpha=0.9)
                ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                ax.set_ylim(1, 7)
                
                # 有意性マーカー（配置手法または時間制限が有意な場合）
                if placement_p < 0.05 or time_p < 0.05:
                    sig_text = []
                    if placement_p < 0.05:
                        sig_text.append(f"配置手法: {'**' if placement_p < 0.01 else '*'}")
                    if time_p < 0.05:
                        sig_text.append(f"時間制限: {'**' if time_p < 0.01 else '*'}")
                    ax.text(0.5, 0.95, ' / '.join(sig_text),
                            transform=ax.transAxes, ha='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                plt.tight_layout()
                
                # ファイル名を安全にする
                safe_filename = f"Q{i:02d}_{q_col[:20].replace('/', '_').replace('。', '')}.png"
                graph_path = os.path.join(graph_dir, safe_filename)
                plt.savefig(graph_path, dpi=300, bbox_inches='tight')
                print(f"\nグラフ保存: {graph_path}")
                plt.close()
                
                # 結果を保存
                anova_results_list.append({
                    '質問番号': f'Q{i}',
                    '質問項目': q_col,
                    '正規性': normality_status,
                    'Shapiro_p': shapiro_p,
                    '検定方法': 'Wilcoxon',
                    '配置手法_p': placement_p,
                    '配置手法_sig': '**' if placement_p < 0.01 else ('*' if placement_p < 0.05 else 'n.s.'),
                    '時間制限_p': time_p,
                    '時間制限_sig': '**' if time_p < 0.01 else ('*' if time_p < 0.05 else 'n.s.'),
                    '交互作用_p': interaction_p,
                    '交互作用_sig': 'N/A',
                })
                
            except Exception as e:
                print(f"⚠️ ウィルコクソン検定エラー: {e}")
                anova_results_list.append({
                    '質問番号': f'Q{i}',
                    '質問項目': q_col,
                    '正規性': normality_status,
                    'Shapiro_p': shapiro_p,
                    '検定方法': 'Wilcoxon',
                    '配置手法_p': np.nan,
                    '配置手法_sig': 'エラー',
                    '時間制限_p': np.nan,
                    '時間制限_sig': 'エラー',
                    '交互作用_p': np.nan,
                    '交互作用_sig': 'エラー',
                })
        
        else:
            print("✓ データは正規分布に従っています（p >= 0.05）")
            print("   → RM-ANOVAを使用します")
            normality_status = "正規"
        
            try:
                # RM-ANOVA実行
                anova_result = AnovaRM(data=df_long, depvar='スコア', subject='被験者ID',
                                       within=['配置手法', '時間制限']).fit()
                
                print(f"\n{anova_result.summary()}")
                
                placement_p = anova_result.anova_table.loc['配置手法', 'Pr > F']
                time_p = anova_result.anova_table.loc['時間制限', 'Pr > F']
                interaction_p = anova_result.anova_table.loc['配置手法:時間制限', 'Pr > F']
                
                print(f"\n配置手法: p={placement_p:.4f} → {'✓ 有意' if placement_p < 0.05 else '× 有意でない'}")
                print(f"時間制限: p={time_p:.4f} → {'✓ 有意' if time_p < 0.05 else '× 有意でない'}")
                print(f"交互作用: p={interaction_p:.4f} → {'✓ 有意' if interaction_p < 0.05 else '× 有意でない'}")
            
                # 記述統計の計算
                interaction_stats = df_long.groupby(['配置手法', '時間制限'])['スコア'].agg(['mean', 'std', 'count'])
                interaction_stats['se'] = interaction_stats['std'] / np.sqrt(interaction_stats['count'])
                
                # グラフ作成
                fig, ax = plt.subplots(figsize=(10, 7))
                
                for method, color, marker in [('fixed', '#e74c3c', 'o'), ('saliency', '#3498db', 's')]:
                    subset = interaction_stats.xs(method, level='配置手法')
                    x_labels = ['時間制限なし', '時間制限あり']
                    means = [subset.loc['なし', 'mean'], subset.loc['あり', 'mean']]
                    errs = [subset.loc['なし', 'se'], subset.loc['あり', 'se']]
                    
                    ax.errorbar(x_labels, means, yerr=errs, 
                                label=f'{method.upper()}配置', 
                                color=color, marker=marker, markersize=12, 
                                linewidth=3, capsize=8, capthick=2)
                
                # タイトルと軸ラベル
                title = f'Q{i}: {q_col[:30]}...' if len(q_col) > 30 else f'Q{i}: {q_col}'
                ax.set_title(f'{title}\n(交互作用 p={interaction_p:.4f})', 
                             fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('時間制限', fontsize=13, fontweight='bold')
                ax.set_ylabel('平均スコア (1-7)', fontsize=13, fontweight='bold')
                ax.legend(fontsize=11, loc='best', framealpha=0.9)
                ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                ax.set_ylim(1, 7)  # リッカート尺度の範囲
                
                # 有意性マーカー
                if interaction_p < 0.05:
                    significance = '**' if interaction_p < 0.01 else '*'
                    ax.text(0.5, 0.95, f'有意な交互作用あり {significance}',
                            transform=ax.transAxes, ha='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                plt.tight_layout()
                
                # ファイル名を安全にする（特殊文字を除去）
                safe_filename = f"Q{i:02d}_{q_col[:20].replace('/', '_').replace('。', '')}.png"
                graph_path = os.path.join(graph_dir, safe_filename)
                plt.savefig(graph_path, dpi=300, bbox_inches='tight')
                print(f"\nグラフ保存: {graph_path}")
                plt.close()
                
                # 結果を保存
                anova_results_list.append({
                    '質問番号': f'Q{i}',
                    '質問項目': q_col,
                    '正規性': normality_status,
                    'Shapiro_p': shapiro_p,
                    '検定方法': 'RM-ANOVA',
                    '配置手法_p': placement_p,
                    '配置手法_sig': '**' if placement_p < 0.01 else ('*' if placement_p < 0.05 else 'n.s.'),
                    '時間制限_p': time_p,
                    '時間制限_sig': '**' if time_p < 0.01 else ('*' if time_p < 0.05 else 'n.s.'),
                    '交互作用_p': interaction_p,
                    '交互作用_sig': '**' if interaction_p < 0.01 else ('*' if interaction_p < 0.05 else 'n.s.'),
                })
            
            except Exception as e:
                print(f"⚠️ エラー: {e}")
                anova_results_list.append({
                    '質問番号': f'Q{i}',
                    '質問項目': q_col,
                    '正規性': normality_status,
                    'Shapiro_p': shapiro_p,
                    '検定方法': 'RM-ANOVA',
                    '配置手法_p': np.nan,
                    '配置手法_sig': 'エラー',
                    '時間制限_p': np.nan,
                    '時間制限_sig': 'エラー',
                    '交互作用_p': np.nan,
                    '交互作用_sig': 'エラー',
                })
    
    # 結果サマリー
    print("\n\n" + "="*80)
    print("=== RM-ANOVA結果サマリー ===")
    print("="*80)
    
    anova_df = pd.DataFrame(anova_results_list)
    print("\n" + anova_df.to_string(index=False))
    
    output_anova = os.path.join('results', 'rm_anova_results.csv')
    anova_df.to_csv(output_anova, encoding='utf-8-sig', index=False)
    print(f"\n\nRM-ANOVA結果を保存: {output_anova}")
    print(f"グラフ保存先: {graph_dir}/")
    
    # 有意な結果のサマリー
    print("\n\n" + "="*80)
    print("=== 有意な効果が見られた項目 ===")
    print("="*80)
    
    print("\n【配置手法による有意差】")
    sig_placement = anova_df[anova_df['配置手法_p'] < 0.05]
    if len(sig_placement) > 0:
        for _, row in sig_placement.iterrows():
            print(f"  {row['配置手法_sig']} {row['質問番号']}: {row['質問項目'][:40]}...")
            print(f"      p={row['配置手法_p']:.4f}")
    else:
        print("  有意な差は見られませんでした")
    
    print("\n【時間制限による有意差】")
    sig_time = anova_df[anova_df['時間制限_p'] < 0.05]
    if len(sig_time) > 0:
        for _, row in sig_time.iterrows():
            print(f"  {row['時間制限_sig']} {row['質問番号']}: {row['質問項目'][:40]}...")
            print(f"      p={row['時間制限_p']:.4f}")
    else:
        print("  有意な差は見られませんでした")
    
    print("\n【交互作用】")
    sig_interaction = anova_df[anova_df['交互作用_p'] < 0.05]
    if len(sig_interaction) > 0:
        for _, row in sig_interaction.iterrows():
            print(f"  {row['交互作用_sig']} {row['質問番号']}: {row['質問項目'][:40]}...")
            print(f"      p={row['交互作用_p']:.4f}")
    else:
        print("  有意な交互作用は見られませんでした")
    
    print("\n" + "="*80)
    print("注: ** p<0.01, * p<0.05, n.s. = not significant")
    print("="*80)


if __name__ == "__main__":
    analyze_questionnaire()