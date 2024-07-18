import os
import pandas as pd
import numpy as np

# Define the path to the folder containing the data files
folder_path = 'data'

# Initialize an empty list to store the data
data = []

# Loop through all files in the folder
for i in range(1609):
    file_path = os.path.join(folder_path, f'fulldata-{}.txt'.format(i))
    
    with open(file_path, 'r') as file:
        # Read the first line and split it into components
        first_line = file.readline().strip().split()
        source_file_name = first_line[0]
        integer1 = int(first_line[1])
        integer2 = int(first_line[2])
        
        # Read the next four lines and convert them to numpy arrays
        array1 = np.array([float(x) for x in file.readline().strip().split()])
        array2 = np.array([float(x) for x in file.readline().strip().split()])
        array3 = np.array([float(x) for x in file.readline().strip().split()])
        array4 = np.array([float(x) for x in file.readline().strip().split()])
        
        # Append the data to the list
        data.append([source_file_name, integer1, integer2, array1, array2, array3, array4])

# Create a pandas DataFrame from the data
df = pd.DataFrame(data, columns=['sourceFileName', 'numberOfNodes', 'numberOfEdges', 'array1', 'array2', 'array3', 'array4'])

# Display the DataFrame
df.to_csv('fullData.csv', sep=',', index=False, encoding='utf-8')