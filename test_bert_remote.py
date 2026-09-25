import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb2.seetacloud.com', 33757, 'root', 'Fpd1YRIp1G35')

def run_cmd(cmd):
    print(f"=== {cmd} ===")
    stdin, stdout, stderr = ssh.exec_command(f"bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && {cmd}'")
    out = stdout.read().decode('utf-8', errors='ignore').strip()
    err = stderr.read().decode('utf-8', errors='ignore').strip()
    if out: print(out)
    if err: print('STDERR:', err)

run_cmd('export HF_ENDPOINT=https://hf-mirror.com && python -c "from transformers import BertModel; m = BertModel.from_pretrained(\'bert-base-uncased\'); print(\'BERT loaded successfully!\', m.__class__.__name__)"')

ssh.close()
