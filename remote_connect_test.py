import paramiko

hostname = "connect.bjb1.seetacloud.com"
port = 26907
username = "root"
password = "Fpd1YRIp1G35"

print(f"Connecting to {username}@{hostname}:{port}...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    client.connect(hostname=hostname, port=port, username=username, password=password, timeout=15)
    print("SSH Connection Successful!")

    def run_remote(cmd):
        print(f"\n>>> Running remote command: {cmd}")
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        if out: print(out.strip())
        if err: print("[STDERR]:", err.strip())
        return out, err

    run_remote("hostname")
    run_remote("nvidia-smi")
    run_remote("python3 --version; which python3")
    run_remote("python3 -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available(), \"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")'")
    run_remote("mkdir -p ~/E_Problem; ls -lh ~/E_Problem; ls -lh ~/")

finally:
    client.close()
    print("\nConnection closed.")
