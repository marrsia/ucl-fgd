import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

def plot_surprisal_lines(csv_path, configs, use_mean=True):
    """
    Plots surprisal across regions for a list of sentence configs.

    Args:
        csv_path: path to surprisal CSV
        configs: list of dicts with keys sentence_group, condition, levels_of_embedding
        use_mean: if True, uses mean surprisal over region; if False, uses first token surprisal
    """
    suffix = '_mean' if use_mean else ''
    ylabel = 'Mean by-word Surprisal in Region' if use_mean else 'First Token Surprisal in Region'

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = cm.tab10(np.linspace(0, 1, len(configs)))

    max_active_labels = []

    for i, config in enumerate(configs):
        row = df[
            (df['sentence_group'] == config['sentence_group']) &
            (df['condition'] == config['condition']) &
            (df['levels_of_embedding'] == config['levels_of_embedding'])
        ]

        if row.empty:
            print(f"Warning: no row found for {config}, skipping")
            continue

        row = row.iloc[0]

        active_columns = [f'main_clause_surprisal{suffix}', f'complementiser_surprisal{suffix}']
        active_labels  = ['main_clause', 'complementiser']

        for lvl in range(1, config['levels_of_embedding'] + 1):
            active_columns.append(f'embedding_{lvl}_surprisal{suffix}')
            active_labels.append(f'embedding_{lvl}')

        active_columns += [f'subject_surprisal{suffix}', f'verb_surprisal{suffix}',
                           f'object_surprisal{suffix}', f'continuation_surprisal{suffix}']
        active_labels  += ['subject', 'verb', 'object', 'continuation']

        surprisals = [row[col] if pd.notna(row[col]) else None for col in active_columns]

        x_vals   = [j for j, s in enumerate(surprisals) if s is not None]
        y_vals   = [s for s in surprisals if s is not None]
        x_labels = [active_labels[j] for j in x_vals]

        label = f"sg{config['sentence_group']} | {config['condition']} | emb{config['levels_of_embedding']}"
        ax.plot(x_vals, y_vals, marker='o', label=label, color=colors[i], linewidth=1.8)

        if len(active_labels) > len(max_active_labels):
            max_active_labels = active_labels

    ax.set_xticks(range(len(max_active_labels)))
    ax.set_xticklabels(max_active_labels, rotation=15, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Region')
    ax.set_title('Surprisal Across Regions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_wh_effect(surprisal_df, sentence_group, embedding_level, condition, surprisal_type):
    """
    Computes the wh-effect for a given sentence group and embedding level.
    
    For gap condition: surprisal(+wh_gap) - surprisal(-wh_gap) at continuation
    For no_gap condition: surprisal(+wh_no_gap) - surprisal(-wh_no_gap) at object
    
    Args:
        surprisal_df: dataframe with surprisal columns
        sentence_group: int
        embedding_level: int
        condition: "gap" or "no_gap"
        surprisal_type: "local" (first token), "semi_local" (object+continuation mean),
                        or "global" (full sentence mean)
    
    Returns:
        wh_effect as float, or None if data missing
    """
    if surprisal_type == 'local':
        gap_col    = 'continuation_surprisal'
        no_gap_col = 'object_surprisal'
    elif surprisal_type == 'semi_local':
        gap_col    = 'semi_local_surprisal_mean'
        no_gap_col = 'semi_local_surprisal_mean'
    elif surprisal_type == 'global':
        gap_col    = 'global_surprisal_mean'
        no_gap_col = 'global_surprisal_mean'
    else:
        raise ValueError(f"surprisal_type must be 'local', 'semi_local' or 'global', got '{surprisal_type}'")

    base = surprisal_df[
        (surprisal_df['sentence_group'] == sentence_group) &
        (surprisal_df['levels_of_embedding'] == embedding_level)
    ]

    if condition == "gap":
        plus_wh  = base[base['condition'] == '+wh_gap'][gap_col].values
        minus_wh = base[base['condition'] == '-wh_gap'][gap_col].values
        if len(plus_wh) == 0 or len(minus_wh) == 0:
            return None
        return plus_wh[0] - minus_wh[0]

    elif condition == "no_gap":
        plus_wh  = base[base['condition'] == '+wh_no_gap'][no_gap_col].values
        minus_wh = base[base['condition'] == '-wh_no_gap'][no_gap_col].values
        if len(plus_wh) == 0 or len(minus_wh) == 0:
            return None
        return plus_wh[0] - minus_wh[0]

    else:
        raise ValueError(f"condition must be 'gap' or 'no_gap', got '{condition}'")


def get_filler_offloading_effect(surprisal_df, sentence_group, embedding_level, condition, surprisal_type):
    """
    Computes the filler offloading effect for a given sentence group and embedding level.
    
    For +wh condition: surprisal(+wh_gap) - surprisal(+wh_no_gap)
    For -wh condition: surprisal(-wh_gap) - surprisal(-wh_no_gap)
    
    Args:
        surprisal_df: dataframe with surprisal columns
        sentence_group: int
        embedding_level: int
        condition: "+wh" or "-wh"
        surprisal_type: "local" (first token of continuation/object), "semi_local" 
                        (object+continuation mean), or "global" (full sentence mean)
    
    Returns:
        filler_offloading_effect as float, or None if data missing
    """
    if surprisal_type == 'local':
        gap_col    = 'continuation_surprisal'
        no_gap_col = 'object_surprisal'
    elif surprisal_type == 'semi_local':
        gap_col    = 'semi_local_surprisal_mean'
        no_gap_col = 'semi_local_surprisal_mean'
    elif surprisal_type == 'global':
        gap_col    = 'global_surprisal_mean'
        no_gap_col = 'global_surprisal_mean'
    else:
        raise ValueError(f"surprisal_type must be 'local', 'semi_local' or 'global', got '{surprisal_type}'")

    base = surprisal_df[
        (surprisal_df['sentence_group'] == sentence_group) &
        (surprisal_df['levels_of_embedding'] == embedding_level)
    ]

    if condition == "+wh":
        gap_row    = base[base['condition'] == '+wh_gap'][gap_col].values
        no_gap_row = base[base['condition'] == '+wh_no_gap'][no_gap_col].values
        if len(gap_row) == 0 or len(no_gap_row) == 0:
            return None
        return gap_row[0] - no_gap_row[0]

    elif condition == "-wh":
        gap_row    = base[base['condition'] == '-wh_gap'][gap_col].values
        no_gap_row = base[base['condition'] == '-wh_no_gap'][no_gap_col].values
        if len(gap_row) == 0 or len(no_gap_row) == 0:
            return None
        return gap_row[0] - no_gap_row[0]

    else:
        raise ValueError(f"condition must be '+wh' or '-wh', got '{condition}'")
      

def mean_ci_95(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    n = len(values)
    if n == 0:
        return None, None
    if n == 1:
        return values[0], 0.0
    mean = np.mean(values)
    ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean, scale=stats.sem(values))
    return mean, (ci_high - mean)  # return half-width to use as yerr
      
  ## TODO 
  # 1. add semi-local filler offloading effect
  # 2. add global suprirsal for filler offloading
  # 3. try to normalise by token frequency???
  

def compute_effect_across_groups(surprisal_df, effect_fn, condition, surprisal_type, embedding_levels=None):
    """
    Computes an effect across all sentence groups and embedding levels.

    Args:
        surprisal_df: dataframe with surprisal columns
        effect_fn: get_wh_effect or get_local_filler_offloading_effect
        condition: condition string to pass to effect_fn
        embedding_levels: list of embedding levels to include, defaults to all

    Returns:
        dict of {embedding_level: [effect values across sentence groups]}
    """
    if embedding_levels is None:
        embedding_levels = sorted(surprisal_df['levels_of_embedding'].unique())

    sentence_groups = sorted(surprisal_df['sentence_group'].unique())

    results = {level: [] for level in embedding_levels}

    for level in embedding_levels:
        for group in sentence_groups:
            effect = effect_fn(surprisal_df, group, level, condition, surprisal_type)
            if effect is not None:
                results[level].append(effect)

    return results


def plot_effect_bar_chart(surprisal_df, effect_fn, conditions, title, ylabel, surprisal_type,
                          condition_labels=None, embedding_levels=None,):
    """
    Plots a grouped bar chart of an effect across embedding levels and conditions.

    Args:
        surprisal_df: dataframe with surprisal columns
        effect_fn: get_wh_effect or get_local_filler_offloading_effect
        conditions: list of condition strings e.g. ["gap", "no_gap"]
        title: plot title
        ylabel: y-axis label
        condition_labels: display names for conditions, defaults to conditions
        embedding_levels: list of embedding levels to include, defaults to all
    """
    if embedding_levels is None:
        embedding_levels = sorted(surprisal_df['levels_of_embedding'].unique())
    if condition_labels is None:
        condition_labels = conditions

    # compute means and CIs for each condition x embedding level
    all_means = {}
    all_cis   = {}
    for condition in conditions:
        results = compute_effect_across_groups(
            surprisal_df, effect_fn, condition, surprisal_type, embedding_levels,
        )
        all_means[condition] = []
        all_cis[condition]   = []
        for level in embedding_levels:
            mean, ci = mean_ci_95(results[level])
            all_means[condition].append(mean)
            all_cis[condition].append(ci)

    # bar chart layout
    n_levels     = len(embedding_levels)
    n_conditions = len(conditions)
    x            = np.arange(n_levels)
    width        = 0.8 / n_conditions
    offsets      = np.linspace(-(n_conditions - 1) / 2, (n_conditions - 1) / 2, n_conditions) * width

    colors = ['#5bbcd6', '#f98400', '#00a08a', '#ff0000'][:n_conditions]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, condition in enumerate(conditions):
        means = all_means[condition]
        cis   = all_cis[condition]
        ax.bar(
            x + offsets[i], means, width,
            yerr=cis, capsize=4,
            label=condition_labels[i],
            color=colors[i], alpha=0.85,
            error_kw={'elinewidth': 1.5}
        )

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f'embedding {l}' for l in embedding_levels])
    ax.set_xlabel('Embedding Level')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
    
def plot_metrics(surprisal_df, surprisal_type='local'):
    """
    Plots wh-effect and filler offloading metrics for each sentence group and embedding level.

    Args:
        surprisal_df: dataframe with surprisal columns
        surprisal_type: "local", "semi_local", or "global"
    """
    sentence_groups  = sorted(surprisal_df['sentence_group'].unique())
    embedding_levels = sorted(surprisal_df['levels_of_embedding'].unique())

    for group_id in sentence_groups:
        n_levels = len(embedding_levels)
        fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 4))

        if n_levels == 1:
            axes = [axes]

        fig.suptitle(f"Sentence Group {group_id}", fontsize=14)

        for ax, level in zip(axes, embedding_levels):
            level_metrics = {
                'wh_effect_gap':         get_wh_effect(surprisal_df, group_id, level, 'gap',    surprisal_type),
                'wh_effect_no_gap':      get_wh_effect(surprisal_df, group_id, level, 'no_gap', surprisal_type),
                'filler_offloading_+wh': get_filler_offloading_effect(surprisal_df, group_id, level, '+wh', surprisal_type),
                'filler_offloading_-wh': get_filler_offloading_effect(surprisal_df, group_id, level, '-wh', surprisal_type),
            }

            labels      = list(level_metrics.keys())
            values      = [v if v is not None else 0 for v in level_metrics.values()]
            colors      = ['#5bbcd6' if 'wh_effect' in l else '#f98400' for l in labels]
            short_labels = [l.replace('wh_effect_', '').replace('filler_offloading_', '') for l in labels]

            x    = np.arange(len(labels))
            bars = ax.bar(x, values, color=colors, alpha=0.85)
            ax.set_title(f"Embedding level {level}")
            ax.set_xticks(x)
            ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=8)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

            # add value labels above/below bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.05 if val >= 0 else -0.15),
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=7.5
                )

        plt.tight_layout()
        plt.show()