import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats


def plot_surprisal_lines(csv_path, configs):
    """
    Plots mean surprisal across regions for a list of sentence configs.

    Args:
        csv_path: path to surprisal CSV
        configs: list of dicts with keys sentence_group, condition, levels_of_embedding
    """
    
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

        active_columns = ['main_clause_surprisal_mean', 'complementiser_surprisal_mean']
        active_labels  = ['main_clause', 'complementiser']

        for lvl in range(1, config['levels_of_embedding'] + 1):
            active_columns.append(f'embedding_{lvl}_surprisal_mean')
            active_labels.append(f'embedding_{lvl}')

        active_columns += ['subject_surprisal_mean', 'verb_surprisal_mean',
                           'object_surprisal_mean', 'continuation_surprisal_mean']
        active_labels  += ['subject', 'verb', 'object', 'continuation']
     
        surprisals = [row[col] if pd.notna(row[col]) else None for col in active_columns]
        
        x_vals  = [j for j, s in enumerate(surprisals) if s is not None]
        y_vals  = [s for s in surprisals if s is not None]
        x_labels = [active_labels[j] for j in x_vals]

        label = f"sg{config['sentence_group']} | {config['condition']} | emb{config['levels_of_embedding']}"
        ax.plot(x_vals, y_vals, marker='o', label=label, color=colors[i], linewidth=1.8)

        if len(active_labels) > len(max_active_labels):
            max_active_labels = active_labels

    ax.set_xticks(range(len(max_active_labels)))
    ax.set_xticklabels(max_active_labels, rotation=15, ha='right')
    ax.set_ylabel('Mean by-word Surprisal in Region')
    ax.set_xlabel('Region')
    ax.set_title('Surprisal Across Regions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_wh_effect(surprisal_df, sentence_group, embedding_level, condition):
    """
    Computes the wh-effect for a given sentence group and embedding level.
    
    For gap condition: surprisal(+wh_gap) - surprisal(-wh_gap) at first token of continuation
    For no_gap condition: surprisal(+wh_no_gap) - surprisal(-wh_no_gap) at first token of object
    
    Args:
        surprisal_df: dataframe with surprisal columns
        sentence_group: int
        embedding_level: int
        condition: "gap" or "no_gap"
    
    Returns:
        wh_effect as float, or None if data missing
    """
    base = surprisal_df[
        (surprisal_df['sentence_group'] == sentence_group) &
        (surprisal_df['levels_of_embedding'] == embedding_level)
    ]

    if condition == "gap":
        plus_wh  = base[base['condition'] == '+wh_gap']['continuation_surprisal'].values
        minus_wh = base[base['condition'] == '-wh_gap']['continuation_surprisal'].values
        if len(plus_wh) == 0 or len(minus_wh) == 0:
            return None
        return plus_wh[0] - minus_wh[0]

    elif condition == "no_gap":
        plus_wh  = base[base['condition'] == '+wh_no_gap']['object_surprisal'].values
        minus_wh = base[base['condition'] == '-wh_no_gap']['object_surprisal'].values
        if len(plus_wh) == 0 or len(minus_wh) == 0:
            return None
        return plus_wh[0] - minus_wh[0]

    else:
        raise ValueError(f"condition must be 'gap' or 'no_gap', got '{condition}'")


def get_local_filler_offloading_effect(surprisal_df, sentence_group, embedding_level, condition):
    """
    Computes the local filler offloading effect for a given sentence group and embedding level.
    
    For +wh condition: surprisal(+wh_gap at continuation) - surprisal(+wh_no_gap at object)
    For -wh condition: surprisal(-wh_gap at continuation) - surprisal(-wh_no_gap at object)
    
    Args:
        surprisal_df: dataframe with surprisal columns
        sentence_group: int
        embedding_level: int
        condition: "+wh" or "-wh"
    
    Returns:
        filler_offloading_effect as float, or None if data missing
    """
    base = surprisal_df[
        (surprisal_df['sentence_group'] == sentence_group) &
        (surprisal_df['levels_of_embedding'] == embedding_level)
    ]

    if condition == "+wh":
        gap_row    = base[base['condition'] == '+wh_gap']['continuation_surprisal'].values
        no_gap_row = base[base['condition'] == '+wh_no_gap']['object_surprisal'].values
        if len(gap_row) == 0 or len(no_gap_row) == 0:
            return None
        return gap_row[0] - no_gap_row[0]

    elif condition == "-wh":
        gap_row    = base[base['condition'] == '-wh_gap']['continuation_surprisal'].values
        no_gap_row = base[base['condition'] == '-wh_no_gap']['object_surprisal'].values
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
  

def compute_effect_across_groups(surprisal_df, effect_fn, condition, embedding_levels=None):
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
            effect = effect_fn(surprisal_df, group, level, condition)
            if effect is not None:
                results[level].append(effect)

    return results


def plot_effect_bar_chart(surprisal_df, effect_fn, conditions, title, ylabel, 
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
            surprisal_df, effect_fn, condition, embedding_levels
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