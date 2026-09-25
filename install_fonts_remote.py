import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb2.seetacloud.com', 33757, 'root', 'Fpd1YRIp1G35')

def run_cmd(cmd):
    print(f"=== {cmd} ===")
    stdin, stdout, stderr = ssh.exec_command(f"bash -lc '{cmd}'")
    out = stdout.read().decode('utf-8', errors='ignore').strip()
    err = stderr.read().decode('utf-8', errors='ignore').strip()
    if out: print(out)
    if err: print('STDERR:', err)

run_cmd('apt-get update -qq && apt-get install -y -qq fonts-wqy-zenhei fonts-wqy-microhei')
run_cmd('fc-list :lang=zh | head -n 5')
ssh.close()
