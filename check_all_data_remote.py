import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb2.seetacloud.com', 33757, 'root', 'Fpd1YRIp1G35')

stdin, stdout, stderr = ssh.exec_command("find /root/E_Problem/DATA -maxdepth 2")
print(stdout.read().decode('utf-8'))

ssh.close()
