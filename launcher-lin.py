import subprocess




command = f"python lin-n-cal-conv-rates.py"
    
# Use subprocess to run the command
process = subprocess.run(command, shell=True)
command_plot = f"python lin-auc-n-cal-conv-rates.py"

# Use subprocess to run the command
process_plot = subprocess.run(command_plot, shell=True)
# Check if the script ran successfully
if process.returncode != 0:
    print(f"Execution failed  n-cal-conv-rates")
else:
    print(f"Execution succeeded n-cal-conv-rates")




command = f"python lin-n-cal.py"
    
# Use subprocess to run the command
process = subprocess.run(command, shell=True)
command_plot = f"python lin-auc-n-cal.py"

# Use subprocess to run the command
process_plot = subprocess.run(command_plot, shell=True)
# Check if the script ran successfully
if process.returncode != 0:
    print(f"Execution failed  n-cal")
else:
    print(f"Execution succeeded n-cal")



command = f"python lin-conv-rate.py"
    
# Use subprocess to run the command
process = subprocess.run(command, shell=True)
command_plot = f"python lin-auc-conv-rate.py"

# Use subprocess to run the command
process_plot = subprocess.run(command_plot, shell=True)
# Check if the script ran successfully
if process.returncode != 0:
    print(f"Execution failed  conv rate")
else:
    print(f"Execution succeeded conv-rate")






command = f"python lin-corr.py"
    
# Use subprocess to run the command
process = subprocess.run(command, shell=True)
command_plot = f"python lin-auc-corr.py"

# Use subprocess to run the command
process_plot = subprocess.run(command_plot, shell=True)
# Check if the script ran successfully
if process.returncode != 0:
    print(f"Execution failed  corr")
else:
    print(f"Execution succeeded corr")




command = f"python snr.py"
    
# Use subprocess to run the command
process = subprocess.run(command, shell=True)
command_plot = f"python auc-snr.py"

# Use subprocess to run the command
process_plot = subprocess.run(command_plot, shell=True)
# Check if the script ran successfully
if process.returncode != 0:
    print(f"Execution failed  snr")
else:
    print(f"Execution succeeded snr")

