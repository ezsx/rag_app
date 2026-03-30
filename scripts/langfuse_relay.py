"""Windows→WSL TCP relay для Langfuse UI (порт 3100).

Docker Desktop с mirrored WSL networking пробрасывает порты только в WSL,
не на Windows host. Этот relay слушает на Windows localhost:3100
и проксирует TCP в WSL localhost:3100 через subprocess pipe.

Тот же паттерн что wsl_tei_relay.py для gpu_server.

Запуск (PowerShell):
    python scripts\\langfuse_relay.py

Langfuse UI: http://localhost:3100
"""

import os
import socket
import subprocess
import sys
import threading

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = int(os.environ.get("LANGFUSE_RELAY_PORT", "3200"))
TARGET_PORT = int(os.environ.get("LANGFUSE_TARGET_PORT", "3100"))
WSL_DISTRO = os.environ.get("WSL_DISTRO", "Ubuntu-22.04")

BUF_SIZE = 65536


def _proxy(client: socket.socket):
    """TCP proxy: Windows socket ↔ WSL socat via subprocess stdin/stdout."""
    try:
        proc = subprocess.Popen(
            [
                "wsl", "-d", WSL_DISTRO, "-e",
                "socat", "-", f"TCP:localhost:{TARGET_PORT}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        # socat не установлен — fallback на bash /dev/tcp
        try:
            proc = subprocess.Popen(
                [
                    "wsl", "-d", WSL_DISTRO, "-e",
                    "bash", "-c",
                    f"exec 3<>/dev/tcp/localhost/{TARGET_PORT}; "
                    "cat <&0 >&3 & cat <&3; kill %1 2>/dev/null",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"  Failed to start WSL proxy: {e}")
            client.close()
            return

    def client_to_wsl():
        try:
            while True:
                data = client.recv(BUF_SIZE)
                if not data:
                    break
                proc.stdin.write(data)
                proc.stdin.flush()
        except Exception:
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    def wsl_to_client():
        try:
            while True:
                data = proc.stdout.read1(BUF_SIZE) if hasattr(proc.stdout, 'read1') else proc.stdout.read(BUF_SIZE)
                if not data:
                    break
                client.sendall(data)
        except Exception:
            pass
        finally:
            client.close()
            proc.terminate()

    t1 = threading.Thread(target=client_to_wsl, daemon=True)
    t2 = threading.Thread(target=wsl_to_client, daemon=True)
    t1.start()
    t2.start()
    t2.join(timeout=120)


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((LISTEN_HOST, LISTEN_PORT))
    srv.listen(32)
    print(f"Langfuse relay: http://{LISTEN_HOST}:{LISTEN_PORT} → WSL localhost:{TARGET_PORT}")
    print("Open http://localhost:3100 in browser")

    try:
        while True:
            client, addr = srv.accept()
            threading.Thread(target=_proxy, args=(client,), daemon=True).start()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
