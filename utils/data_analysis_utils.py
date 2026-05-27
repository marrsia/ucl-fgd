import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import math
from scipy import stats

def plot_surprisal_lines(csv_path, configs, use_mean=True, save_path=None):
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

        label = f"sg{config['condition']} | emb{config['levels_of_embedding']}"
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def get_wh_effect(surprisal_df, sentence_group, embedding_level, condition, surprisal_type, gap_type):
    """
    Computes the wh-effect for a given sentence group and embedding level.
    
    For gap condition: surprisal(+wh_gap) - surprisal(-wh_gap) at continuation
    For no_gap condition: surprisal(+wh_no_gap) - surprisal(-wh_no_gap) at object
    
    Args:
        surprisal_df: dataframe with surprisal columns
        sentence_group: int
        embedding_level: int
        condition: "gap" or "no_gap"
        surprisal_type: "local" (first token), "semi_local" (from gap position to the end of the sentence),
                        or "global" (full sentence mean)
        gap_type: "object" or "subject"
    
    Returns:
        wh_effect as float, or None if data missing
    """
    if surprisal_type == 'local':
        if gap_type == 'object':
            gap_col    = 'continuation_surprisal'
            no_gap_col = 'object_surprisal'
        elif gap_type == 'subject':
            gap_col = 'verb_surprisal'
            no_gap_col = 'subject_surprisal'
        else:
            raise ValueError(f"gap_type must be 'object' or 'subject', got {gap_type}")
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


def get_filler_offloading_effect(surprisal_df, sentence_group, embedding_level, condition, surprisal_type, gap_type):
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
        if gap_type == 'object':
            gap_col    = 'continuation_surprisal'
            no_gap_col = 'object_surprisal'
        elif gap_type == 'subject':
            gap_col = 'verb_surprisal'
            no_gap_col = 'subject_surprisal'
        else:
            raise ValueError(f"gap_type must be 'object' or 'subject', got {gap_type}")
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
  

def compute_effect_across_groups(surprisal_df, effect_fn, condition, surprisal_type, gap_type, embedding_levels=None):
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
            effect = effect_fn(surprisal_df, group, level, condition, surprisal_type, gap_type)
            if effect is not None:
                results[level].append(effect)

    return results


def plot_effect_bar_chart(surprisal_df, effect_fn, conditions, title, ylabel, surprisal_type, gap_type, 
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
            surprisal_df, effect_fn, condition, surprisal_type, gap_type, embedding_levels
        )
        all_means[condition] = []
        all_cis[condition]   = []
        for level in embedding_levels:
            mean, ci = mean_ci_95(results[level])
            all_means[condition].append(mean if mean is not None else 0)
            all_cis[condition].append(ci if ci is not None else 0)

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
    
    print(f"\n=== STANDARD DEVIATIONS ({title}) ===")
    for condition in conditions:
        print(f"\n  {condition_labels[conditions.index(condition)]}:")
        for j, level in enumerate(embedding_levels):
            values = compute_effect_across_groups(
                surprisal_df, effect_fn, condition, surprisal_type, gap_type, [level]
            )[level]
            if len(values) > 0:
                print(f"    embedding {level}: mean={np.mean(values):.3f}, sd={np.std(values, ddof=1):.3f}, n={len(values)}")
            else:
                print(f"    embedding {level}: no data")

    plt.tight_layout()
    plt.show()
    
def plot_effect_bar_chart_multi_model(
    model_dfs, effect_fn, conditions, title, ylabel, surprisal_type, gap_type,
    condition_labels=None, embedding_levels=None, save_path=None):
    """
    Plots a grouped bar chart of an effect across embedding levels and models
    on a single plot. One color per model; conditions differentiated by hatch pattern.
    Within each embedding level, bars are grouped by model, with conditions side by side.

    Args:
        model_dfs: dict of {model_name: surprisal_df}, ordered (e.g. GPT-2, GRNN, ngram)
        effect_fn: get_wh_effect or get_local_filler_offloading_effect
        conditions: list of condition strings e.g. ["gap", "no_gap"]
        title: plot title
        ylabel: y-axis label
        condition_labels: display names for conditions, defaults to conditions
        embedding_levels: list of embedding levels, defaults to all (from first df)
    """
    if condition_labels is None:
        condition_labels = conditions

    first_df = next(iter(model_dfs.values()))
    if embedding_levels is None:
        embedding_levels = sorted(first_df['levels_of_embedding'].unique())

    n_levels     = len(embedding_levels)
    n_models     = len(model_dfs)
    n_conditions = len(conditions)
    n_bars       = n_models * n_conditions  # bars per embedding level

    # one color per model
    model_colors = ['#5bbcd6', '#f98400', '#00a08a', '#9b59b6'][:n_models]
    # second condition gets hatching; first condition is solid
    hatches = ['', 'o'][:n_conditions]

    x       = np.arange(n_levels)
    width   = 0.8 / n_bars
    # center the whole group of bars around each x tick
    offsets = np.arange(n_bars) * width - (n_bars - 1) * width / 2

    fig, ax = plt.subplots(figsize=(11, 5))

    # precompute all data and track y extents for limits
    all_vals = []
    bar_idx  = 0
    legend_handles = []

    for m_idx, (model_name, surprisal_df) in enumerate(model_dfs.items()):
        color = model_colors[m_idx]

        for c_idx, condition in enumerate(conditions):
            results = compute_effect_across_groups(
                surprisal_df, effect_fn, condition, surprisal_type, gap_type, embedding_levels
            )
            means, cis = [], []
            for level in embedding_levels:
                mean, ci = mean_ci_95(results[level])
                
                means.append(mean if (mean is not None and not math.isnan(mean)) else 0.0)
                cis.append(ci   if (ci   is not None and not math.isnan(ci))   else 0.0)
        
                if mean is not None and not math.isnan(mean):
                    all_vals.extend([mean - ci, mean + ci])

            alpha = 0.85 if hatches[c_idx] == '' else 0.5
            bars = ax.bar(
                x + offsets[bar_idx], means, width,
                yerr=cis, capsize=3,
                color=color, alpha=alpha,
                hatch=hatches[c_idx],
                edgecolor='white' if hatches[c_idx] == '' else 'black',
                linewidth=0.5,
                error_kw={'elinewidth': 1.5}
            )
            # one legend entry per model (first condition only), one for hatch pattern
            if c_idx == 0:
                legend_handles.append(
                    mpatches.Patch(facecolor=color, alpha=0.85, label=model_name)
                )
            if m_idx == 0:
                legend_handles.append(
                    mpatches.Patch(
                        facecolor='grey', alpha=0.5,
                        hatch=hatches[c_idx],
                        edgecolor='grey',
                        label=condition_labels[c_idx]
                    )
                )

            bar_idx += 1

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Embedding {l}' for l in embedding_levels])
    ax.set_xlabel('Embedding Level')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(handles=legend_handles, framealpha=0.9, 
          bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    ax.grid(True, alpha=0.3, axis='y')

    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        margin = (hi - lo) * 0.15 or 0.5
        ax.set_ylim(lo - margin, hi + margin)

    plt.tight_layout()

    print(f"\n=== STANDARD DEVIATIONS ({title}) ===")
    for model_name, surprisal_df in model_dfs.items():
        print(f"\n  [{model_name}]")
        for condition, label in zip(conditions, condition_labels):
            print(f"  {label}:")
            for level in embedding_levels:
                values = compute_effect_across_groups(
                    surprisal_df, effect_fn, condition, surprisal_type, gap_type, [level]
                )[level]
                if values:
                    print(f"    embedding {level}: mean={np.mean(values):.3f}, "
                          f"sd={np.std(values, ddof=1):.3f}, n={len(values)}")
                else:
                    print(f"    embedding {level}: no data")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_metrics(surprisal_df, gap_type, surprisal_type='local'):
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
                'wh_effect_gap':         get_wh_effect(surprisal_df, group_id, level, 'gap',    surprisal_type, gap_type),
                'wh_effect_no_gap':      get_wh_effect(surprisal_df, group_id, level, 'no_gap', surprisal_type, gap_type),
                'filler_offloading_+wh': get_filler_offloading_effect(surprisal_df, group_id, level, '+wh', surprisal_type, gap_type),
                'filler_offloading_-wh': get_filler_offloading_effect(surprisal_df, group_id, level, '-wh', surprisal_type, gap_type),
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
        

def plot_metrics(surprisal_df, gap_type, surprisal_type='local'):
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
                'wh_effect_gap':         get_wh_effect(surprisal_df, group_id, level, 'gap',    surprisal_type, gap_type),
                'wh_effect_no_gap':      get_wh_effect(surprisal_df, group_id, level, 'no_gap', surprisal_type, gap_type),
                'filler_offloading_+wh': get_filler_offloading_effect(surprisal_df, group_id, level, '+wh', surprisal_type, gap_type),
                'filler_offloading_-wh': get_filler_offloading_effect(surprisal_df, group_id, level, '-wh', surprisal_type, gap_type),
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

def compute_average_surprisals_by_condition(csv_path, regions, surprisal_type='mean'):
    """
    Reads a surprisal CSV and computes average surprisal of given regions by condition.

    Args:
        csv_path: path to surprisal CSV
        regions: list of region names e.g. ['complementiser', 'object', 'continuation']
        surprisal_type: 'mean' (region mean) or 'local' (first token)

    Returns:
        dataframe of average surprisal per condition per region
    """
    df = pd.read_csv(csv_path)

    suffix = '_surprisal_mean' if surprisal_type == 'mean' else '_surprisal'

    cols = [f'{region}{suffix}' for region in regions]

    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}")

    result = df.groupby('condition')[cols].mean().round(3)
    result.columns = regions

    return result