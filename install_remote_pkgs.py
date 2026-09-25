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

run_cmd('curl -I -s --connect-timeout 5 https://www.baidu.com | head -n 3')
run_cmd('pip install scikit-learn transformers opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple')

ssh.close()
