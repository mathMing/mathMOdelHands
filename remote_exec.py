import sys
import paramiko

hostname = "connect.bjb2.seetacloud.com"
port = 33757
username = "root"
password = "Fpd1YRIp1G35"

if len(sys.argv) < 2:
    print("Usage: python remote_exec.py <command>")
    sys.exit(1)

cmd = " ".join(sys.argv[1:])

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=hostname, port=port, username=username, password=password, timeout=30)

try:
    # Use a remote script wrapper or direct execution
    exec_script = f"""source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/E_Problem
{cmd}
"""
    # Write to a temporary file on remote and execute
    sftp = ssh.open_sftp()
    with sftp.file('/tmp/run_task.sh', 'w') as f:
        f.write(exec_script)
    sftp.chmod('/tmp/run_task.sh', 0o755)
    sftp.close()

    print(f"=== Remote Executing via /tmp/run_task.sh: ===")
    print(exec_script.strip())
    print("==============================================")
    
    stdin, stdout, stderr = ssh.exec_command("bash /tmp/run_task.sh", get_pty=True)
    
    for line in iter(stdout.readline, ""):
        print(line, end="", flush=True)
    
    exit_status = stdout.channel.recv_exit_status()
    print(f"\n=== Process Exited with Code: {exit_status} ===")
    sys.exit(exit_status)
finally:
    ssh.close()
