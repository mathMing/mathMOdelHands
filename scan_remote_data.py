import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb2.seetacloud.com', 33757, 'root', 'Fpd1YRIp1G35')

sftp = ssh.open_sftp()
sftp.put(r"X:\mathModelHands\scan_files.py", "/tmp/scan_files.py")
sftp.close()

stdin, stdout, stderr = ssh.exec_command("bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && python /tmp/scan_files.py'")
print(stdout.read().decode('utf-8'))
print(stderr.read().decode('utf-8'))
ssh.close()
