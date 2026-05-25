import pandas as pd

def rename_conditions(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    #df = df.rename(columns={'gap': 'wh', 'filler': 'gap'})
    
    #df['gap'] = df['gap'].map({'yes': 'no', 'no': 'yes'})
    
    condition_map = {
        '+wh_no_gap' : "+wh_gap",
        '-wh_gap' : "-wh_no_gap"
    }
    df['condition'] = df['condition'].map(condition_map)
    
    df.to_csv(output_csv, index=False)
    #print(df[['wh', 'gap', 'condition']].drop_duplicates().sort_values('condition'))
    
rename_conditions("data/model_outputs/predictions/continuations_to_label.csv", "data/model_outputs/predictions/continuations_to_label.csv")