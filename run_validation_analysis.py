import os
import sys
import paramiko

hostname = "connect.bjb2.seetacloud.com"
port = 33757
username = "root"
password = "Fpd1YRIp1G35"

print("Connecting to remote server...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname, port, username, password)

sftp = ssh.open_sftp()
local_script = r"X:\mathModelHands\problem2_robust\evaluate_validation_and_error_analysis.py"
remote_script = "/root/E_Problem/problem2_robust/evaluate_validation_and_error_analysis.py"
print(f"Uploading {local_script} to {remote_script}...")
sftp.put(local_script, remote_script)

print("Executing validation evaluation and error analysis on remote 5090...")
cmd = "bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && cd /root/E_Problem && python problem2_robust/evaluate_validation_and_error_analysis.py'"
stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)

for line in iter(stdout.readline, ""):
    print(line, end="", flush=True)

status = stdout.channel.recv_exit_status()
print(f"Execution finished with code {status}")

if status == 0:
    print("Downloading generated figures and summary JSON...")
    files_to_download = [
        ("/root/E_Problem/problem2_robust/figures/fig8_validation_confusion_matrix.png", r"X:\mathModelHands\problem2_robust\figures\fig8_validation_confusion_matrix.png"),
        ("/root/E_Problem/problem2_robust/figures/fig9_validation_regression_residuals.png", r"X:\mathModelHands\problem2_robust\figures\fig9_validation_regression_residuals.png"),
        ("/root/E_Problem/problem2_robust/figures/fig10_global_modality_gate_distribution.png", r"X:\mathModelHands\problem2_robust\figures\fig10_global_modality_gate_distribution.png"),
        ("/root/E_Problem/problem2_robust/validation_error_attribution_summary.json", r"X:\mathModelHands\problem2_robust\validation_error_attribution_summary.json")
    ]
    for rem, loc in files_to_download:
        print(f"Downloading {rem} -> {loc}")
        sftp.get(rem, loc)
    print("All validation evaluation files downloaded successfully!")

sftp.close()
ssh.close()
