import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 乱数シードの固定（研究の再現性確保のため）
np.random.seed(42)

N = 1000000       # エージェント総数
T = 40          # シミュレーション期間（40年間：22歳〜62歳を想定）

# ==============================================================================
# 【最新若者学歴比率 × リアル労働統計】パラメータ定義（5クラス統合版）
# ==============================================================================
# ShortTermHigher: 専門学校卒(13.0%) と 短大高専卒(13.1%) を合算（計 26.1%）
edu_config = {
    'Compulsory':      {'ratio': 0.008, 'init_wealth': 0.6, 'base_income': 2.0, 'luck_bonus': 0.5, 'p_misfortune': 0.20, 'misfortune_damage': 1.0, 'growth': -0.002, 'trap': True},
    'HighSchool':      {'ratio': 0.145, 'init_wealth': 1.0, 'base_income': 2.5, 'luck_bonus': 0.8, 'p_misfortune': 0.15, 'misfortune_damage': 0.8, 'growth':  0.000, 'trap': True},
    'ShortTermHigher': {'ratio': 0.261, 'init_wealth': 1.2, 'base_income': 2.8, 'luck_bonus': 1.0, 'p_misfortune': 0.12, 'misfortune_damage': 0.7, 'growth':  0.001, 'trap': False},
    'University':      {'ratio': 0.512, 'init_wealth': 2.0, 'base_income': 3.8, 'luck_bonus': 1.5, 'p_misfortune': 0.09, 'misfortune_damage': 0.6, 'growth':  0.003, 'trap': False},
    'GradSchool':      {'ratio': 0.074, 'init_wealth': 3.0, 'base_income': 4.5, 'luck_bonus': 2.0, 'p_misfortune': 0.08, 'misfortune_damage': 0.5, 'growth':  0.005, 'trap': False}
}

edu_names = list(edu_config.keys())
edu_ratios = [edu_config[e]['ratio'] for e in edu_names]

# ==============================================================================
# 1. 静的初期化フェーズ
# ==============================================================================
agent_edu = np.random.choice(edu_names, size=N, p=edu_ratios)

# 初期才能（平均0.6, 標準偏差0.1の正規分布、[0.0, 1.0]にクリップ）
initial_talent = np.clip(np.random.normal(loc=0.6, scale=0.1, size=N), 0.0, 1.0)

df = pd.DataFrame({
    'education': agent_edu,
    'initial_talent': initial_talent,
    'talent': initial_talent.copy(),
    'wealth': [edu_config[e]['init_wealth'] for e in agent_edu],
    'luck_success': 0,
    'misfortune_count': 0
})

# ==============================================================================
# 2. 動的時間発展フェーズ（40年ループ・政府介入なし）
# ==============================================================================
for year in range(T):
    mean_wealth = df['wealth'].mean() if df['wealth'].mean() > 0 else 1.0
    
    # ベクトル演算用の配列マッピング
    base_incomes = np.array([edu_config[e]['base_income'] for e in df['education']])
    luck_bonuses = np.array([edu_config[e]['luck_bonus'] for e in df['education']])
    p_misfortunes = np.array([edu_config[e]['p_misfortune'] for e in df['education']])
    misfortune_damages = np.array([edu_config[e]['misfortune_damage'] for e in df['education']])
    growths = np.array([edu_config[e]['growth'] for e in df['education']])
    traps = np.array([edu_config[e]['trap'] for e in df['education']])
    
    # ① 基礎収入の追加と生活費・税金の消費（収入の60% + 資産の5%）
    living_cost = (base_incomes * 0.60) + (df['wealth'] * 0.05)
    df['wealth'] = df['wealth'] + base_incomes - living_cost
    
    # ② 実力の経年変化（成長・陳腐化）
    df['talent'] = np.clip(df['talent'] + growths, 0.0, 1.0)
    
    # ③ 累積的優位（動的マタイ効果）：資産が社会平均を上回る割合に応じて確率ブースト
    p_luck = 0.15 + np.clip((df['wealth'] / mean_wealth) * 0.01, 0.0, 0.05)
    
    # ④ 幸運の判定（確率 p_luck に遭遇し、かつ乱数が実力以下でボーナス獲得）
    luck_success = (np.random.rand(N) < p_luck) & (np.random.rand(N) <= df['talent'])
    bonus_gain = luck_bonuses * df['talent']
    df['wealth'] = np.where(luck_success, df['wealth'] + bonus_gain, df['wealth'])
    df['luck_success'] += luck_success.astype(int)
    
    # ⑤ 不運の判定と構造的排除
    misfortune_encountered = np.random.rand(N) < p_misfortunes
    df['wealth'] = np.where(misfortune_encountered, df['wealth'] - misfortune_damages, df['wealth'])
    df['wealth'] = np.clip(df['wealth'], 0.1, None) # 最低生活セーフティネット
    df['misfortune_count'] += misfortune_encountered.astype(int)
    
    # 階層転落トラップ（義務教育卒・高卒かつ不運遭遇で翌年以降の才能に直接ペナルティ）
    trap_trigger = misfortune_encountered & traps
    df['talent'] = np.clip(df['talent'] - np.where(trap_trigger, 0.02, 0.0), 0.0, 1.0)

df['final_talent'] = df['talent']

# ==============================================================================
# 3. 統計分析・詳細レポート出力
# ==============================================================================
print("======================================================================")
print(" 社会学・数理経済学ABM 研究発表用詳細レポート（5クラス統合モデル）")
print("======================================================================")

# 最終資産の対数変換（目的変数）
df['log_wealth'] = np.log10(df['wealth'])

# 学歴ダミー変数の作成（HighSchoolを基準=0にするためドロップ）
edu_dummies = pd.get_dummies(df['education'], dtype=float)
X_dummy = edu_dummies.drop(columns=['HighSchool'])

# 説明変数の結合
X = pd.concat([
    df[['initial_talent', 'final_talent', 'luck_success', 'misfortune_count']],
    X_dummy
], axis=1)
y = df['log_wealth']

# 多重共線性を考慮した標準化および線形回帰分析
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
reg = LinearRegression().fit(X_scaled, y)

print(f"■ モデル決定係数 (R²): {reg.score(X_scaled, y):.4f}\n")

print("■ 標準化偏回帰分析（影響度ランキング Beta値）:")
beta_df = pd.DataFrame({'Factor': X.columns, 'Beta': reg.coef_})
beta_df['Abs_Beta'] = beta_df['Beta'].abs()
beta_df = beta_df.sort_values(by='Abs_Beta', ascending=False).drop(columns=['Abs_Beta'])

for idx, row in enumerate(beta_df.itertuples(), start=1):
    print(f"  {idx}位: {row.Factor:<18} | Beta: {row.Beta: .4f}")

# 学歴別の最終資産【平均値】と【相対値（高卒=1.0基準）】
mean_wealth_absolute = df.groupby('education')['wealth'].mean().reindex(edu_names)
highschool_mean = mean_wealth_absolute['HighSchool']
mean_wealth_relative = mean_wealth_absolute / highschool_mean

print("\n■ 学歴別 最終資産平均値（実額および高卒基準の相対倍率）:")
for edu in edu_names:
    abs_val = mean_wealth_absolute[edu]
    rel_val = mean_wealth_relative[edu]
    print(f"  - {edu:<16} : 実額 {abs_val:5.2f} | 相対値 {rel_val:.3f} 倍 (高卒={highschool_mean:.2f})")

print(f"\n★ 院卒(GradSchool) と 義務教育卒(Compulsory) の構造的格差倍率: {mean_wealth_relative['GradSchool'] / mean_wealth_relative['Compulsory']:.2f} 倍")

print("\n■ 最終資産トップ3 保有者個別プロファイル（ケーススタディ用）:")
top3 = df.sort_values(by='wealth', ascending=False).head(3)
for idx, row in enumerate(top3.itertuples(), start=1):
    print(f"  【第{idx}位】")
    print(f"    学歴: {row.education} | 初期才能: {row.initial_talent:.3f} | 最終実力: {row.final_talent:.3f}")
    print(f"    幸運成功: {row.luck_success}回 | 不運遭遇: {row.misfortune_count}回 | 最終資産額: {row.wealth:.2f}")

# ==============================================================================
# 4. 可視化出力（指定の2つのグラフを左右に並べて表示）
# ==============================================================================
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo', 'Arial']
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 6色パレットから5色を選択（統合されたクラスに対応）
colors = sns.color_palette('Set2', n_colors=5)
edu_color_map = dict(zip(edu_names, colors))

# --- グラフ①：資産順ソート能力散布図 ---
df_sorted = df.sort_values(by='wealth').reset_index(drop=True)
df_sorted['agent_rank'] = df_sorted.index

sns.scatterplot(
    data=df_sorted,
    x='agent_rank',
    y='final_talent',
    hue='education',
    hue_order=edu_names,
    palette=edu_color_map,
    alpha=0.6,
    edgecolor=None,
    s=15,
    ax=axes[0]
)
axes[0].set_title('グラフ①: 資産順ソート能力散布図（純粋な能力値分布）', fontsize=13, fontweight='bold')
axes[0].set_xlabel('資産順位 (Agent Rank: 0 〜 9999)', fontsize=11)
axes[0].set_ylabel('現在の能力値 (0.0 〜 1.0)', fontsize=11)
axes[0].legend(title='学歴区分', loc='upper left')

# --- グラフ②：学歴別最終資産【相対値】棒グラフ ---
barplot = sns.barplot(
    x=mean_wealth_relative.index,
    y=mean_wealth_relative.values,
    palette=edu_color_map,
    hue=mean_wealth_relative.index,
    legend=False,
    ax=axes[1]
)
axes[1].set_title('グラフ②: 学歴別最終資産平均の相対値（高卒 = 1.0 基準）', fontsize=13, fontweight='bold')
axes[1].set_xlabel('学歴区分', fontsize=11)
axes[1].set_ylabel('最終資産の相対値（格差倍率）', fontsize=11)
axes[1].set_xticklabels(mean_wealth_relative.index, rotation=15)
axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)  # 高卒の基準線

# 各棒の上に具体的な数値を太字で表示
for p in barplot.patches:
    height = p.get_height()
    if height > 0:
        axes[1].annotate(
            f'{height:.2f}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            xytext=(0, 3),
            textcoords='offset points'
        )

plt.tight_layout()
plt.show()