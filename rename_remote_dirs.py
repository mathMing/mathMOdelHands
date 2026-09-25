import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb2.seetacloud.com', 33757, 'root', 'Fpd1YRIp1G35')

script = """
import os
import glob

# 1. Att3 rename
d3 = "/root/E_Problem/DATA/Att3-Incomplete-Modality-Features"
for item in os.listdir(d3):
    full = os.path.join(d3, item)
    if os.path.isdir(full):
        if "δ" in item:
            tgt = os.path.join(d3, "未对齐版本")
            if full != tgt:
                os.rename(full, tgt)
                print(f"Renamed Att3: {item} -> 未对齐版本")
        else:
            tgt = os.path.join(d3, "对齐版本")
            if full != tgt:
                os.rename(full, tgt)
                print(f"Renamed Att3: {item} -> 对齐版本")

# 2. Att4 rename
d4 = "/root/E_Problem/DATA/Att4-Interpretable-Video-Samples-Features"
for item in os.listdir(d4):
    full = os.path.join(d4, item)
    if os.path.isdir(full) and item != "附件4-可解释专项视频样本与特征文件":
        tgt = os.path.join(d4, "附件4-可解释专项视频样本与特征文件")
        os.rename(full, tgt)
        print(f"Renamed Att4 base: {item} -> 附件4-可解释专项视频样本与特征文件")

sub4 = os.path.join(d4, "附件4-可解释专项视频样本与特征文件")
for item in os.listdir(sub4):
    full = os.path.join(sub4, item)
    if os.path.isdir(full):
        if "δ" in item:
            tgt = os.path.join(sub4, "未对齐版本")
            if full != tgt:
                os.rename(full, tgt)
                print(f"Renamed Att4 inner: {item} -> 未对齐版本")
        elif item != "未对齐版本" and item != "对齐版本":
            tgt = os.path.join(sub4, "对齐版本")
            if full != tgt:
                os.rename(full, tgt)
                print(f"Renamed Att4 inner: {item} -> 对齐版本")

# Verify
att3_aligned_pkls = glob.glob("/root/E_Problem/DATA/Att3-Incomplete-Modality-Features/对齐版本/*.pkl")
att4_aligned_pkls = glob.glob("/root/E_Problem/DATA/Att4-Interpretable-Video-Samples-Features/附件4-可解释专项视频样本与特征文件/对齐版本/*.pkl")
print("Verified Att3 aligned pkl count:", len(att3_aligned_pkls))
print("Verified Att4 aligned pkl count:", len(att4_aligned_pkls))
"""

sftp = ssh.open_sftp()
with sftp.file('/tmp/rename_clean.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && python /tmp/rename_clean.py'")
print(stdout.read().decode('utf-8'))
print(stderr.read().decode('utf-8'))
ssh.close()
