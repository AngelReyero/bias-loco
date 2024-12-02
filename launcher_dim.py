import subprocess

# Define the different values for y_method
y_methods = ['nonlin', 'nonlin2', 'poly'] 

# Loop through the y_method values and run code.py
for method in y_methods:
    # Construct the command to run code.py with the current y_method
    command = f"python dimension.py --y_method {method}"
    
    # Use subprocess to run the command
    process = subprocess.run(command, shell=True)
    command_plot = f"python auc-dim.py --y_method {method}"
    
    # Use subprocess to run the command
    process_plot = subprocess.run(command_plot, shell=True)
    # Check if the script ran successfully
    if process.returncode != 0:
        print(f"Execution failed for y_method: {method}")
    else:
        print(f"Execution succeeded for y_method: {method}")
