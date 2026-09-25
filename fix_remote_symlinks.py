import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb2.seetacloud.com', 33757, 'root', 'Fpd1YRIp1G35')

script = """
import os
import glob

# 1. Att3 symlinks
att3_dir = "/root/E_Problem/DATA/Att3-Incomplete-Modality-Features"
subdirs = [os.path.join(att3_dir, d) for d in os.listdir(att3_dir) if os.path.isdir(os.path.join(att3_dir, d))]
print("Att3 subdirs:", subdirs)

# Distinguish aligned vs unaligned in Att3
for d in subdirs:
    # Check sample file
    pkls = glob.glob(os.path.join(d, "*.pkl"))
    if pkls:
        # Check if first pkl filename has 'aligned' or check content
        import pickle
        with open(pkls[0], 'rb') as f:
            data = pickle.load(f)['test']
            is_aligned = data['text'].shape[1] == 50 if 'text' in data else False
            t_len = data['text'].shape[1] if 'text' in data else 'N/A'
            print(f"Dir: {d}, sample: {pkls[0]}, text len: {t_len}, is_aligned: {is_aligned}")
            
            target_link = os.path.join(att3_dir, "对齐版本" if is_aligned else "未对齐版本")
            if not os.path.exists(target_link):
                os.symlink(d, target_link)
                print(f"Created symlink: {target_link} -> {d}")

# 2. Att4 symlinks
att4_dir = "/root/E_Problem/DATA/Att4-Interpretable-Video-Samples-Features"
att4_sub = [os.path.join(att4_dir, d) for d in os.listdir(att4_dir) if os.path.isdir(os.path.join(att4_dir, d))][0]
print("Att4 root inner:", att4_sub)

# Ensure '附件4-可解释专项视频样本与特征文件' link exists
att4_link_base = os.path.join(att4_dir, "附件4-可解释专项视频样本与特征文件")
if not os.path.exists(att4_link_base):
    os.symlink(att4_sub, att4_link_base)
    print(f"Created symlink: {att4_link_base} -> {att4_sub}")

inner_subdirs = [os.path.join(att4_sub, d) for d in os.listdir(att4_sub) if os.path.isdir(os.path.join(att4_sub, d))]
print("Att4 inner subdirs:", inner_subdirs)
for d in inner_subdirs:
    pkls = glob.glob(os.path.join(d, "*.pkl"))
    if pkls:
        import pickle
        with open(pkls[0], 'rb') as f:
            data = pickle.load(f)['test']
            t_len = data['text'].shape[1] if 'text' in data else 'N/A'
            is_aligned = (t_len == 50)
            print(f"Att4 Dir: {d}, sample: {pkls[0]}, text len: {t_len}, is_aligned: {is_aligned}")
            target_link = os.path.join(att4_sub, "对齐版本" if is_aligned else "未对齐版本")
            if not os.path.exists(target_link):
                os.symlink(d, target_link)
                print(f"Created symlink: {target_link} -> {d}")

print("Symlinks setup finished!")
"""

sftp = ssh.open_sftp()
with sftp.file('/tmp/setup_symlinks.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = ssh.exec_command("bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && python /tmp/setup_symlinks.py'")
print(stdout.read().decode('utf-8'))
print(stderr.read().decode('utf-8'))

ssh.close()
