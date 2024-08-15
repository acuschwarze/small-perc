# aggregate data from 2d parameter sweep for finite theory and simulations

import os
import numpy as np

def aggregate_data(num_nodes, attack_flag, finite=True):
    base_dir = "data/synthetic_data"
    result_dir = "data/synthetic_data"
    
    # Prepare the result array
    result_array = np.zeros((100, num_nodes))

    for i in range(1, 101):
        probability = i / 100
        subfolder = os.path.join(base_dir, f"p{probability:.2f}")
        
        if not os.path.exists(subfolder):
            continue

        if finite:
            fstring = "relSCurve"
        else:
            fstring = f"infRelSCurve"

        filename = f"{fstring}_attack{attack_flag}_n{num_nodes}_p{probability:.2f}.npy"
        file_path = os.path.join(subfolder, filename)
        
        if not os.path.isfile(file_path):
            continue
        
        # Load the .npy file and append to result_array
        data = np.load(file_path)
        result_array[i-1] = data
    
    
    # Save the aggregated array
    result_filename = f"{fstring}_attack{attack_flag}_n{num_nodes}.npy"
    result_filepath = os.path.join(result_dir, result_filename)
    np.save(result_filepath, result_array)
    print(f"Aggregated data saved to {result_filepath}")

def aggregate_data_2d(num_nodes, attack_flag, num_trials=100):
    base_dir = "data/synthetic_data"
    result_dir = "data/synthetic_data"
    
    # Prepare the result array
    result_array = np.zeros((100, num_nodes,num_trials))

    for i in range(1, 101):
        probability = i / 100
        subfolder = os.path.join(base_dir, f"p{probability:.2f}")
        
        if not os.path.exists(subfolder):
            continue

        #filename = f"simRelSCurve_attack{attack_flag}_n{num_nodes}_p{probability:.2f}.npy"
        filename = f"simRelSCurve{num_trials}_attack{attack_flag}_n{num_nodes}_p{probability:.2f}.npy"
        file_path = os.path.join(subfolder, filename)
        
        if not os.path.isfile(file_path):
            continue
        
        # Load the .npy file and append to result_array
        data = np.load(file_path)

        # remove first iteration because it is empty?
        #data = data[1:]
        if i==1:
            print(data[0])
        result_array[i-1,:,:] = data.T 
    
    
    # Save the aggregated array
    #result_filename = f"simRelSCurve_attack{attack_flag}_n{num_nodes}.npy"
    result_filename = f"simRelSCurve{num_trials}_attack{attack_flag}_n{num_nodes}.npy"
    result_filepath = os.path.join(result_dir, result_filename)
    np.save(result_filepath, result_array)
    print(f"Aggregated data saved to {result_filepath}")

if True:
    for i in range(1,101):
        aggregate_data_2d(i, True, num_trials=100)
        aggregate_data_2d(i, False, num_trials=100)

if False:
    for i in range(1,101):
        aggregate_data(i, True)
        aggregate_data(i, False)

if False:
    for i in range(1,101):
        aggregate_data(i, True, finite=False)
        aggregate_data(i, False, finite=False)

