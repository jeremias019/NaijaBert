# Load the preprocessed dataset
data_path = 'path to labelled.csv'
data = pd.read_csv(data_path)

# Clean trailing spaces in labels
data['Label'] = data['Label'].str.strip()

# Map labels to integers
label_mapping = {'LABEL 0': 0, 'LABEL 1': 1, 'LABEL 2': 2}
data['Label'] = data['Label'].map(label_mapping)

# Split dataset
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
