import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb2.seetacloud.com', 33757, 'root', 'Fpd1YRIp1G35')

script = """
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import BertModel, BertTokenizer
print("Downloading / verifying bert-base-uncased from HF mirror...")
model = BertModel.from_pretrained('bert-base-uncased')
print("BertModel loaded successfully!", model.__class__.__name__)
"""

sftp = ssh.open_sftp()
with sftp.file('/tmp/download_bert.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && python /tmp/download_bert.py'", get_pty=True)
for line in iter(stdout.readline, ""):
    print(line, end="", flush=True)

ssh.close()
