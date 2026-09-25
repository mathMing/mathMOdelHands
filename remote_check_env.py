import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.bjb1.seetacloud.com', 26907, 'root', 'Fpd1YRIp1G35', timeout=15)

def run_cmd(cmd):
    stdin, stdout, stderr = client.exec_command(f"bash -lc '{cmd}'")
    print(f"=== {cmd} ===")
    out = stdout.read().decode('utf-8', errors='ignore').strip()
    err = stderr.read().decode('utf-8', errors='ignore').strip()
    if out: print(out)
    if err: print('ERR:', err)

run_cmd('source /root/miniconda3/etc/profile.d/conda.sh && conda env list')
run_cmd('which python; python --version')
run_cmd('python -c "import torch; print(\\"PyTorch:\\", torch.__version__, \\"CUDA:\\", torch.cuda.is_available(), \\"Device:\\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\"None\\")"')
run_cmd('ls -la /root/E_Problem/DATA')
client.close()
