import os
import sys
import paramiko
from stat import S_ISDIR

hostname = "connect.bjb1.seetacloud.com"
port = 26907
username = "root"
password = "Fpd1YRIp1G35"

local_base = r"X:\mathModelHands"
remote_base = "/root/E_Problem"

# Folders and files to upload
items_to_upload = [
    "problem1_unaligned",
    "problem2_robust",
    "problem3_interpretability",
    "problem.pdf"
]

print(f"Connecting to {username}@{hostname}:{port}...")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=hostname, port=port, username=username, password=password, timeout=30)
print("Connected! Initializing SFTP...")

sftp = ssh.open_sftp()

def ensure_remote_dir(remote_dir):
    dirs = []
    current = remote_dir
    while current and current != "/":
        dirs.append(current)
        current = os.path.dirname(current)
    for d in reversed(dirs):
        try:
            sftp.stat(d)
        except IOError:
            try:
                sftp.mkdir(d)
                print(f"Created remote dir: {d}")
            except Exception as e:
                pass

def upload_file_with_progress(local_path, remote_path):
    file_size = os.path.getsize(local_path)
    def callback(transferred, total):
        pass
    sftp.put(local_path, remote_path, callback=callback)

def upload_dir(local_dir, remote_dir):
    ensure_remote_dir(remote_dir)
    for root, dirs, files in os.walk(local_dir):
        # Ignore unwanted directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.idea', '.vscode']]
        
        rel_path = os.path.relpath(root, local_dir)
        cur_remote_dir = remote_dir if rel_path == "." else os.path.join(remote_dir, rel_path).replace("\\", "/")
        ensure_remote_dir(cur_remote_dir)
        
        for file in files:
            if file.endswith('.pyc') or file.startswith('.DS_Store'):
                continue
            local_file = os.path.join(root, file)
            remote_file = os.path.join(cur_remote_dir, file).replace("\\", "/")
            
            # Check if remote file exists and has same size
            try:
                rem_stat = sftp.stat(remote_file)
                if rem_stat.st_size == os.path.getsize(local_file):
                    # print(f"Skipping identical: {rel_path}/{file}")
                    continue
            except IOError:
                pass
            
            print(f"Uploading: {rel_path}/{file} ({os.path.getsize(local_file)/1024:.1f} KB)...")
            upload_file_with_progress(local_file, remote_file)

try:
    ensure_remote_dir(remote_base)
    for item in items_to_upload:
        local_path = os.path.join(local_base, item)
        remote_path = f"{remote_base}/{item}"
        if os.path.isfile(local_path):
            print(f"Uploading file {item}...")
            upload_file_with_progress(local_path, remote_path)
            print(f"Uploaded {item}.")
        elif os.path.isdir(local_path):
            print(f"\n--- Uploading directory {item} ---")
            upload_dir(local_path, remote_path)
            print(f"--- Finished directory {item} ---")
    
    print("\nAll files uploaded successfully!")

    # Verify remote structure
    stdin, stdout, stderr = ssh.exec_command(f"ls -lh {remote_base}")
    print("\nRemote directory contents:")
    print(stdout.read().decode('utf-8'))
    
    stdin, stdout, stderr = ssh.exec_command(f"ls -lh {remote_base}/problem2_robust")
    print("\nproblem2_robust remote contents:")
    print(stdout.read().decode('utf-8'))

finally:
    sftp.close()
    ssh.close()
    print("Connection closed.")
