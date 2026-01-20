import pandas as pd
import numpy as np


# Load your behavioral data
path = 'data_in_csv/state_space/21_set_raw.csv'

# Define which features are "internal states" vs "behavioral outputs"
internal_states = ['Sleep Quality',
'Waking Energy',
'Dominant Emotion', 
'Dominant Emotion intensity', 
 ] # or whatever you tracked

behavioral_features = [
    'Work',
'Focused Learning',
'Skill practicing',
'Physical Endeavors',
'Scrolling',
'jorking',
'Passive media',
'Active Media',
'System Architecture']

substance_features = ['NIC',
'CAF',
'ALC',
'L-TY']



def find_correlations_s_I(ds_path: str, substances: list , internal_states: list) -> np.array:
    df = pd.read_csv(ds_path)
    subset = df[substances + internal_states]
    # Correlation matrix for all these columns
    corr_full = subset.corr()
    # m × n block: behavioral_features (rows) vs internal_states (cols)
    A_corr = corr_full.loc[substances, internal_states].to_numpy()

    return A_corr


def find_correlations_i_b(ds_path: str, behavioral_features: list , internal_states: list) -> np.array:
    df = pd.read_csv(ds_path)
    subset = df[behavioral_features + internal_states]
    # Correlation matrix for all these columns
    corr_full = subset.corr()
    # m × n block: behavioral_features (rows) vs internal_states (cols)
    A_corr = corr_full.loc[behavioral_features, internal_states].to_numpy()

    return A_corr
        

def get_interaction_matrix(raw_correlations,threshold=0.15,):
    """Discretizes raw correlations into {-1, 0, 1}."""
    A = np.zeros_like(raw_correlations)
    A[raw_correlations >= threshold] = 1
    A[raw_correlations <= -threshold] = -1
    return A




if __name__ == "__main__":
    s_corr = find_correlations_s_I(path,internal_states,substance_features)
    enhanced_interaction = get_interaction_matrix(s_corr)
    np.savetxt("interaction_matrices/internal+substances",enhanced_interaction )
    print(f' Internal state with substances interaction matrix \n {enhanced_interaction} \n')
    correlations = find_correlations_i_b(path, behavioral_features,internal_states)

    interaction_matrix = get_interaction_matrix(correlations)
    print(f' Internal state with behavioral interaction matrix \n\n {interaction_matrix}')
    np.savetxt("interaction_matrices/behavioral+internal",interaction_matrix)