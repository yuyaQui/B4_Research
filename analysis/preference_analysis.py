import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 日本語フォント設定
rcParams['font.family'] = 'MS Gothic'
rcParams['axes.unicode_minus'] = False

def analyze_final_questionnaire():
    """
    最終アンケート（配置手法の好み）を集計・解析
    """
    input_path = os.path.join('results', '結果（主観指標） - アンケート（最後）.csv')
    
    if not os.path.exists(input_path):
        print(f"⚠️ ファイルが見つかりません: {input_path}")
        return
    
    print("="*80)
    print("最終アンケート集計・解析")
    print("="*80)
    print(f"\nデータ読み込み: {input_path}")
    
    # データ読み込み
    df = pd.read_csv(input_path)
    print(f"有効回答数: {len(df)}")
    
    # 質問列を取得（4列目以降、最後のコメント列を除く）
    question_columns = df.columns[4:-1].tolist()
    
    print(f"\n質問項目数: {len(question_columns)}")
    print("\n--- 質問項目 ---")
    for i, q in enumerate(question_columns, 1):
        print(f"Q{i}: {q.strip()}")
    
    # 集計結果を格納
    results = []
    
    print("\n\n" + "="*80)
    print("質問ごとの集計結果")
    print("="*80)
    
    for i, q_col in enumerate(question_columns, 1):
        print(f"\n【Q{i}】{q_col.strip()}")
        print("-" * 80)
        
        # 回答の集計
        value_counts = df[q_col].value_counts()
        
        # 配置A, B, どちらでもないの数をカウント
        count_a = value_counts.get('配置 A（動的配置）', 0)
        count_b = value_counts.get('配置 B（固定配置）', 0)
        count_neither = value_counts.get('どちらでもない', 0)
        total = count_a + count_b + count_neither
        
        # パーセンテージ計算
        pct_a = (count_a / total * 100) if total > 0 else 0
        pct_b = (count_b / total * 100) if total > 0 else 0
        pct_neither = (count_neither / total * 100) if total > 0 else 0
        
        print(f"  配置A（動的配置）: {count_a}人 ({pct_a:.1f}%)")
        print(f"  配置B（固定配置）: {count_b}人 ({pct_b:.1f}%)")
        print(f"  どちらでもない   : {count_neither}人 ({pct_neither:.1f}%)")
        print(f"  合計             : {total}人")
        
        # 結果を保存
        results.append({
            '質問番号': f'Q{i}',
            '質問項目': q_col.strip(),
            '配置A（動的）': count_a,
            '配置B（固定）': count_b,
            'どちらでもない': count_neither,
            '配置A割合(%)': pct_a,
            '配置B割合(%)': pct_b,
            'どちらでもない割合(%)': pct_neither,
            '優位な配置': '配置A' if count_a > count_b else ('配置B' if count_b > count_a else '同数')
        })
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)
    
    # 結果を保存
    output_path = os.path.join('results', 'final_questionnaire_summary.csv')
    results_df.to_csv(output_path, encoding='utf-8-sig', index=False)
    print(f"\n\n集計結果を保存: {output_path}")
    
    # サマリー表示
    print("\n\n" + "="*80)
    print("集計サマリー")
    print("="*80)
    print("\n" + results_df[['質問番号', '質問項目', '配置A（動的）', '配置B（固定）', 'どちらでもない', '優位な配置']].to_string(index=False))
    
    # 全体的な傾向
    print("\n\n" + "="*80)
    print("全体的な傾向")
    print("="*80)
    
    total_a = results_df['配置A（動的）'].sum()
    total_b = results_df['配置B（固定）'].sum()
    total_neither = results_df['どちらでもない'].sum()
    grand_total = total_a + total_b + total_neither
    
    print(f"\n全質問の合計回答数:")
    print(f"  配置A（動的配置）: {total_a}回答 ({total_a/grand_total*100:.1f}%)")
    print(f"  配置B（固定配置）: {total_b}回答 ({total_b/grand_total*100:.1f}%)")
    print(f"  どちらでもない   : {total_neither}回答 ({total_neither/grand_total*100:.1f}%)")
    
    # 配置Aが優位な質問数
    a_wins = len(results_df[results_df['優位な配置'] == '配置A'])
    b_wins = len(results_df[results_df['優位な配置'] == '配置B'])
    ties = len(results_df[results_df['優位な配置'] == '同数'])
    
    print(f"\n質問ごとの優位性:")
    print(f"  配置Aが優位: {a_wins}問")
    print(f"  配置Bが優位: {b_wins}問")
    print(f"  同数        : {ties}問")
    
    # グラフ作成
    create_visualizations(results_df, question_columns)
    
    # 自由記述コメントの表示
    print("\n\n" + "="*80)
    print("自由記述コメント")
    print("="*80)
    
    comment_col = df.columns[-1]
    for idx, row in df.iterrows():
        comment = row[comment_col]
        if pd.notna(comment) and str(comment).strip():
            print(f"\n【被験者{row['実験番号']}】")
            print(f"  {comment}")


def create_visualizations(results_df, question_columns):
    """
    可視化グラフを作成
    """
    graph_dir = os.path.join('results', 'final_questionnaire_graphs')
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    # グラフ1: 質問ごとの回答分布（積み上げ棒グラフ）
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(results_df))
    width = 0.6
    
    # データ準備
    a_counts = results_df['配置A（動的）'].values
    b_counts = results_df['配置B（固定）'].values
    neither_counts = results_df['どちらでもない'].values
    
    # 積み上げ棒グラフ
    p1 = ax.bar(x, a_counts, width, label='配置A（動的配置）', color='#3498db', alpha=0.8)
    p2 = ax.bar(x, b_counts, width, bottom=a_counts, label='配置B（固定配置）', color='#e74c3c', alpha=0.8)
    p3 = ax.bar(x, neither_counts, width, bottom=a_counts+b_counts, label='どちらでもない', color='#95a5a6', alpha=0.8)
    
    ax.set_xlabel('質問項目', fontsize=12, fontweight='bold')
    ax.set_ylabel('回答数', fontsize=12, fontweight='bold')
    ax.set_title('最終アンケート: 質問ごとの配置手法の好み', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{i+1}' for i in range(len(results_df))], rotation=0)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    graph_path = os.path.join(graph_dir, 'question_distribution.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"\nグラフ保存: {graph_path}")
    plt.close()
    
    # 質問ごとの配置A vs B の比較（横棒グラフ）
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y = np.arange(len(results_df))
    
    # 配置Aを正、配置Bを負として表示
    a_counts = results_df['配置A（動的）'].values
    b_counts = -results_df['配置B（固定）'].values  # 負にする
    
    ax.barh(y, a_counts, height=0.7, label='配置A（動的配置）', color='#3498db', alpha=0.8)
    ax.barh(y, b_counts, height=0.7, label='配置B（固定配置）', color='#e74c3c', alpha=0.8)
    
    ax.set_yticks(y)
    # 質問を短縮表示
    short_labels = [f"Q{i+1}: {q[:25]}..." if len(q) > 25 else f"Q{i+1}: {q}" 
                    for i, q in enumerate(results_df['質問項目'])]
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel('回答数（左: 配置B、右: 配置A）', fontsize=12, fontweight='bold')
    ax.set_title('配置手法の好み比較（質問別）', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.invert_yaxis()  # Y軸を反転して上から下に表示
    
    plt.tight_layout()
    graph_path = os.path.join(graph_dir, 'preference_comparison.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"グラフ保存: {graph_path}")
    plt.close()
    
    print(f"\nすべてのグラフを保存しました: {graph_dir}/")


if __name__ == "__main__":
    analyze_final_questionnaire()
