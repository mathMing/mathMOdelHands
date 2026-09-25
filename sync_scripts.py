import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb2.seetacloud.com', 33757, 'root', 'Fpd1YRIp1G35')

sftp = ssh.open_sftp()

files_to_sync = [
    (r"X:\mathModelHands\problem2_robust\model.py", "/root/E_Problem/problem2_robust/model.py"),
    (r"X:\mathModelHands\problem2_robust\train.py", "/root/E_Problem/problem2_robust/train.py"),
    (r"X:\mathModelHands\problem2_robust\ablation_experiments.py", "/root/E_Problem/problem2_robust/ablation_experiments.py"),
    (r"X:\mathModelHands\problem2_robust\predict_att3.py", "/root/E_Problem/problem2_robust/predict_att3.py"),
    (r"X:\mathModelHands\problem3_interpretability\predict_att4_interpretable.py", "/root/E_Problem/problem3_interpretability/predict_att4_interpretable.py"),
    (r"X:\mathModelHands\problem3_interpretability\generate_explanation_cards.py", "/root/E_Problem/problem3_interpretability/generate_explanation_cards.py"),
]

for local_path, remote_path in files_to_sync:
    print(f"Uploading {local_path} -> {remote_path}")
    sftp.put(local_path, remote_path)

print("All scripts synced successfully!")
sftp.close()
ssh.close()
