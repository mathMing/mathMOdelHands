import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb2.seetacloud.com', 33757, 'root', 'Fpd1YRIp1G35')

script = """
import os

# Att3
d3 = "/root/E_Problem/DATA/Att3-Incomplete-Modality-Features"
for item in os.listdir(d3):
    full = os.path.join(d3, item)
    if os.path.isdir(full):
        if "δ" in item:
            link = os.path.join(d3, "未对齐版本")
            if os.path.islink(link) or os.path.exists(link): os.unlink(link)
            os.symlink(full, link)
            print(f"Linked Att3 unaligned: {link} -> {full}")
        else:
            link = os.path.join(d3, "对齐版本")
            if os.path.islink(link) or os.path.exists(link): os.unlink(link)
            os.symlink(full, link)
            print(f"Linked Att3 aligned: {link} -> {full}")

# Att4
d4 = "/root/E_Problem/DATA/Att4-Interpretable-Video-Samples-Features"
sub4 = [os.path.join(d4, x) for x in os.listdir(d4) if os.path.isdir(os.path.join(d4, x))][0]
link4_base = os.path.join(d4, "附件4-可解释专项视频样本与特征文件")
if os.path.islink(link4_base) or os.path.exists(link4_base): os.unlink(link4_base)
os.symlink(sub4, link4_base)
print(f"Linked Att4 base: {link4_base} -> {sub4}")

for item in os.listdir(sub4):
    full = os.path.join(sub4, item)
    if os.path.isdir(full):
        if "δ" in item:
            link = os.path.join(sub4, "未对齐版本")
            if os.path.islink(link) or os.path.exists(link): os.unlink(link)
            os.symlink(full, link)
            print(f"Linked Att4 unaligned: {link} -> {full}")
        elif "汾" in item or "" in item:
            link = os.path.join(sub4, "对齐版本")
            if os.path.islink(link) or os.path.exists(link): os.unlink(link)
            os.symlink(full, link)
            print(f"Linked Att4 aligned: {link} -> {full}")

# Test access
import glob
print("Test Att3 aligned files count:", len(glob.glob("/root/E_Problem/DATA/Att3-Incomplete-Modality-Features/对齐版本/*.pkl")))
print("Test Att4 aligned files count:", len(glob.glob("/root/E_Problem/DATA/Att4-Interpretable-Video-Samples-Features/附件4-可解释专项视频样本与特征文件/对齐版本/*.pkl")))
"""

sftp = ssh.open_sftp()
with sftp.file('/tmp/link_all.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && python /tmp/link_all.py'")
print(stdout.read().decode('utf-8'))
print(stderr.read().decode('utf-8'))
ssh.close()
