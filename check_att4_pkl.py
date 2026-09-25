import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb2.seetacloud.com', 33757, 'root', 'Fpd1YRIp1G35')

script = """
import pickle
import os
import glob

att4_base = '/root/E_Problem/DATA/Att4-Interpretable-Video-Samples-Features'
for root, dirs, files in os.walk(att4_base):
    for f in files:
        if f.endswith('.pkl'):
            p = os.path.join(root, f)
            with open(p, 'rb') as fp:
                obj = pickle.load(fp)
                print(p, 'keys:', type(obj), list(obj.keys()) if isinstance(obj, dict) else 'Not a dict')
            break
"""

sftp = ssh.open_sftp()
with sftp.file('/tmp/check_att4_pkl.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && python /tmp/check_att4_pkl.py'")
print(stdout.read().decode('utf-8'))
print(stderr.read().decode('utf-8'))
ssh.close()
