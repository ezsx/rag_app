# Docker ↔ WSL2 networking: the definitive fix

**The root cause is architectural: Docker containers and WSL2 distros live in completely isolated network namespaces with no routing between them.** `host.docker.internal` resolves to Docker's internal vpnkit IP (192.168.65.2), and while vpnkit reconnects to Windows `localhost`, WSL2's automatic localhost forwarding is unreliable and often fails to bridge the gap. The fastest production-grade fix is **`netsh interface portproxy`**, which creates a genuine Windows-level TCP listener on port 8082 that Docker containers can reach via `host.docker.internal:8082`. The long-term cleanest solution is containerizing `gpu_server.py` with NVIDIA GPU passthrough, eliminating the cross-network problem entirely.

---

## Why `host.docker.internal:8082` silently fails

The failure stems from a **three-VM architecture** that Docker Desktop on WSL2 creates. Understanding the full network topology is essential:

```
┌─────────────────────────────────────────────────────────────────────┐
│  WINDOWS HOST                                                       │
│                                                                     │
│  llama-server.exe (:8080)  ← bound to 0.0.0.0 on Windows ✅        │
│                                                                     │
│  ┌────────────────────┐        ┌──────────────────────────┐         │
│  │ vEthernet (WSL)    │        │ vpnkit (com.docker.      │         │
│  │ 172.28.0.1         │        │ backend) shared-memory   │         │
│  └──────┬─────────────┘        │ 192.168.65.0/24          │         │
│         │                      └──────────┬───────────────┘         │
│  ┌──────┴─────────────┐        ┌──────────┴───────────────┐         │
│  │ WSL2 VM (Ubuntu)   │        │ WSL2 VM (docker-desktop) │         │
│  │ eth0: 172.28.x.x   │        │ eth0: 192.168.65.3       │         │
│  │                    │        │  ┌─────────────────────┐ │         │
│  │ gpu_server.py      │        │  │ Docker bridge       │ │         │
│  │ 0.0.0.0:8082       │        │  │ 172.17.0.0/16       │ │         │
│  │                    │        │  │  ┌───────────────┐  │ │         │
│  └────────────────────┘        │  │  │ rag_app       │  │ │         │
│         ↑ NO ROUTE             │  │  │ 172.17.0.2    │  │ │         │
│                                │  │  └───────────────┘  │ │         │
│  wslrelay.exe tries to        │  └─────────────────────┘ │         │
│  forward to Windows localhost  └──────────────────────────┘         │
│  (unreliable)                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

When a container calls `host.docker.internal:8080`, vpnkit intercepts the traffic on its 192.168.65.2 virtual IP, then `com.docker.backend` makes a native Windows `connect()` call to `localhost:8080`. Since `llama-server.exe` is a Windows-native process bound to `0.0.0.0:8080`, this connects directly. **Port 8080 works because the service runs natively on Windows.**

For port 8082, the chain adds a fragile extra hop: vpnkit → Windows `localhost:8082` → `wslrelay.exe` → WSL2 VM → `gpu_server.py`. The `wslrelay.exe` mechanism that auto-forwards WSL2 ports to Windows `localhost` is **asynchronous, undocumented, and unreliable**. It polls for bound ports inside WSL2 and creates relay entries, but this frequently breaks after network changes, WSL restarts, or simply due to timing issues. Multiple GitHub issues (microsoft/WSL#6364, #6530, #12023) document this behavior. The forwarding only exposes ports on `127.0.0.1` — not on any other Windows interface — and even that mapping is inconsistent.

**The 172.28.x.x (WSL2) and 172.17.x.x (Docker bridge) subnets are completely isolated.** There is no routing between them. Docker containers cannot reach WSL2's IP directly, and WSL2 cannot reach Docker container IPs directly. The only bridge is through the Windows host via vpnkit.

---

## Why the relay returns 404 on `/rerank` and `/colbert-encode`

The 404 errors on the Windows relay (`:18082`) are almost certainly **not a networking issue** — they indicate an application-level problem. A TCP-level relay (like `netsh portproxy` or a simple TCP forwarder) is path-agnostic: it forwards raw bytes without inspecting HTTP content. If `/embed` returns 200 but `/rerank` returns 404, the 404 is coming from `gpu_server.py` itself, not from the relay.

Three likely causes deserve investigation:

- **Path mismatch**: If `gpu_server.py` registers routes at `/v1/rerank` and `/v1/colbert-encode` but the client sends requests to `/rerank` and `/colbert-encode` (without the `/v1/` prefix), the server returns 404. Verify exact route registrations in `gpu_server.py` against the client's configured paths.
- **Incomplete endpoint registration**: If `gpu_server.py` is a custom FastAPI server, the `/rerank` and `/colbert-encode` routes may have conditional registration that fails silently — for example, a missing model file or a failed import that prevents those routes from being added at startup.
- **HTTP method mismatch**: `/embed` may accept both GET and POST, while `/rerank` requires POST with a specific JSON body. A relay that changes methods or drops the request body would trigger 404.

**Diagnostic step** — run this directly inside WSL2 to isolate the issue:
```bash
# From inside WSL2 (not through any relay)
curl -v -X POST http://localhost:8082/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"test","documents":["doc1","doc2"]}'

curl -v -X POST http://localhost:8082/colbert-encode \
  -H "Content-Type: application/json" \
  -d '{"input":["test"]}'

# Check registered FastAPI routes
curl http://localhost:8082/openapi.json | python3 -m json.tool | grep path
```

If these also 404 from inside WSL2, the problem is in `gpu_server.py`'s route configuration, not networking.

---

## Seven options compared head-to-head

| Solution | Works? | Latency | Reboot-safe | Complexity | Stable endpoint | Verdict |
|---|---|---|---|---|---|---|
| **A) `netsh portproxy`** | ✅ Yes | ~0ms (L4 kernel) | ⚠️ Script needed | Medium | `host.docker.internal:8082` | **★★★★★ Best quick fix** |
| **B) Reverse proxy (Caddy/nginx)** | ✅ Yes | ~1ms (L7) | ✅ Windows service | Medium | `host.docker.internal:808x` | ★★★★ Good, port conflict risk |
| **C) Direct WSL2 IP** | ❌ Isolated subnets | N/A | ❌ IP changes | High | None | ★ Not viable |
| **D) socat/SSH tunnel** | ⚠️ Partial | Low–Medium | ⚠️ Fragile | High | ⚠️ | ★★ Niche only |
| **E) Proxy container in Docker** | ⚠️ Needs A or F | +0.5ms hop | Depends | Medium | Docker service name | ★★★ Good complement |
| **F) WSL2 mirrored networking** | ⚠️ Buggy | ~0ms | ✅ Config only | Low | `host.docker.internal:8082` | ★★★ Promising but risky |
| **G) Containerize gpu_server** | ✅ Yes | 0ms (same network) | ✅ Docker manages | Medium initial | `gpu-server:8082` | **★★★★★ Best long-term** |

**Option A (`netsh portproxy`) is the recommended immediate fix.** It operates at the Windows kernel's TCP layer with negligible latency overhead, requires no additional software, and is the approach Microsoft officially documents. The only maintenance burden is updating the WSL2 IP on reboot via a scheduled script.

**Option G (containerize gpu_server.py)** is the best long-term architecture. NVIDIA Container Toolkit fully supports GPU passthrough on WSL2 with Docker Desktop — this eliminates the entire cross-network problem by putting everything on the same Docker bridge network.

**Option F (mirrored networking) is not production-ready.** Despite Docker Desktop 4.26.0+ adding support, multiple confirmed bugs remain: TCP connection stalls inside containers (moby/moby#48201), Docker port forwarding failures (microsoft/WSL#10494), and Docker Desktop startup crashes requiring `nestedVirtualization=true` (docker/for-win#14691). Avoid this until the ecosystem stabilizes.

**Option C (direct WSL2 IP) does not work** — the 172.28.x.x (WSL2) and 172.17.x.x (Docker bridge) subnets have no routing between them. Docker containers cannot reach WSL2's IP address.

---

## Primary fix: `netsh interface portproxy` with automation

### How it solves the problem

`netsh interface portproxy` creates a Windows kernel-level TCP forwarder that listens on **all Windows interfaces** (`0.0.0.0:8082`) and relays connections to the WSL2 VM's IP on port 8082. This means Windows is now genuinely listening on port 8082 — not relying on the flaky `wslrelay.exe` mechanism. When Docker's vpnkit connects to `host.docker.internal:8082`, it hits the real Windows listener, which forwards to WSL2. The connection chain becomes deterministic:

```
Container → 192.168.65.2:8082 (vpnkit) → com.docker.backend
  → connect(0.0.0.0:8082) → netsh portproxy → WSL2 172.28.x.x:8082
  → gpu_server.py ✅
```

### Step 1 — Manual setup (PowerShell as Administrator)

```powershell
# Get WSL2 IP
$wslIP = (wsl hostname -I).Trim().Split(" ")[0]
Write-Host "WSL2 IP: $wslIP"

# Create port proxy rule
netsh interface portproxy add v4tov4 `
  listenaddress=0.0.0.0 listenport=8082 `
  connectaddress=$wslIP connectport=8082

# Create Windows Firewall inbound rule
New-NetFireWallRule -DisplayName 'WSL2-GPU-Server-8082' `
  -Direction Inbound -LocalPort 8082 -Action Allow -Protocol TCP

# On Windows 11 22H2+ with Hyper-V firewall, also run:
Set-NetFirewallHyperVVMSetting `
  -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
  -DefaultInboundAction Allow

# Verify
netsh interface portproxy show v4tov4
```

### Step 2 — Production automation script

The portproxy rule itself persists across reboots (stored in Windows registry), **but WSL2's IP changes on every reboot** in NAT mode, so the rule points to a stale address. This script handles automatic updates:

**File: `C:\Scripts\WSL2-PortForward.ps1`**
```powershell
# Requires running as Administrator
param([int[]]$Ports = @(8082))

Start-Transcript -Path "C:\Logs\WSL2-PortForward.log" -Append

# Ensure WSL is running (may not be ready at logon)
$retries = 0
do {
    $wslIP = (wsl -d Ubuntu hostname -I 2>$null).Trim().Split(" ")[0]
    if ($wslIP -notmatch '^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$') {
        Write-Host "$(Get-Date) - WSL not ready, retrying in 5s..."
        Start-Sleep -Seconds 5
        $retries++
    }
} while ($wslIP -notmatch '^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$' -and $retries -lt 12)

if ($retries -ge 12) {
    Write-Host "ERROR: Could not get WSL2 IP after 60s"
    Stop-Transcript; exit 1
}

Write-Host "$(Get-Date) - WSL2 IP: $wslIP"

foreach ($port in $Ports) {
    netsh interface portproxy delete v4tov4 `
      listenport=$port listenaddress=0.0.0.0 2>$null
    netsh interface portproxy add v4tov4 `
      listenport=$port listenaddress=0.0.0.0 `
      connectport=$port connectaddress=$wslIP
    Write-Host "$(Get-Date) - Forwarding 0.0.0.0:$port -> ${wslIP}:${port}"
}

# Ensure firewall rules exist
Remove-NetFireWallRule -DisplayName 'WSL2-GPU-Ports' -ErrorAction SilentlyContinue
New-NetFireWallRule -DisplayName 'WSL2-GPU-Ports' `
  -Direction Inbound -LocalPort $Ports -Action Allow -Protocol TCP

netsh interface portproxy show v4tov4
Stop-Transcript
```

### Step 3 — Task Scheduler registration

```powershell
$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File C:\Scripts\WSL2-PortForward.ps1"
$trigger = New-ScheduledTaskTrigger -AtLogon
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal `
  -GroupId "BUILTIN\Administrators" -RunLevel Highest

Register-ScheduledTask -TaskName "WSL2PortForward" `
  -Action $action -Trigger $trigger `
  -Settings $settings -Principal $principal
```

### Step 4 — Docker Compose configuration

```yaml
services:
  rag_app:
    # ... existing config ...
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - LLAMA_SERVER_URL=http://host.docker.internal:8080    # Windows V100
      - GPU_SERVER_URL=http://host.docker.internal:8082      # WSL2 RTX 5060 Ti
      # Remove the old relay port reference
      # - GPU_SERVER_URL=http://host.docker.internal:18082   # ← DELETE THIS

  qdrant:
    # ... existing config (unchanged) ...
```

### Step 5 — Verification from inside Docker

```bash
# Test all three endpoints
docker exec -it rag_app curl -s -o /dev/null -w "%{http_code}" \
  http://host.docker.internal:8082/embed
# Expected: 200

docker exec -it rag_app curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://host.docker.internal:8082/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"test","documents":["a","b"]}'
# Expected: 200

docker exec -it rag_app curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://host.docker.internal:8082/colbert-encode \
  -H "Content-Type: application/json" \
  -d '{"input":["test"]}'
# Expected: 200

# Verify DNS resolution
docker exec -it rag_app nslookup host.docker.internal
# Expected: 192.168.65.2

# Verify llama-server still works
docker exec -it rag_app curl -s http://host.docker.internal:8080/health
# Expected: 200
```

### Rollback plan

```powershell
# Remove portproxy rule
netsh interface portproxy delete v4tov4 listenport=8082 listenaddress=0.0.0.0

# Remove firewall rule
Remove-NetFireWallRule -DisplayName 'WSL2-GPU-Ports'

# Remove scheduled task
Unregister-ScheduledTask -TaskName "WSL2PortForward" -Confirm:$false

# Revert Docker Compose to use old relay
# GPU_SERVER_URL=http://host.docker.internal:18082
```

---

## Backup path: containerize gpu_server.py with GPU passthrough

If the portproxy approach proves insufficient — or for a cleaner long-term architecture — moving `gpu_server.py` into Docker eliminates the networking problem entirely. **NVIDIA Container Toolkit supports full GPU passthrough on WSL2 with Docker Desktop**, confirmed by NVIDIA, Docker, and Microsoft documentation.

```dockerfile
# gpu_server/Dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY gpu_server.py .
EXPOSE 8082
CMD ["python3", "gpu_server.py", "--host", "0.0.0.0", "--port", "8082"]
```

```yaml
# docker-compose.yml addition
services:
  gpu-server:
    build: ./gpu_server
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']   # RTX 5060 Ti (second GPU)
              capabilities: [gpu]
    ports:
      - "8082:8082"
    networks:
      - app-network

  rag_app:
    environment:
      - GPU_SERVER_URL=http://gpu-server:8082     # Docker service name — no hacks
      - LLAMA_SERVER_URL=http://host.docker.internal:8080
    networks:
      - app-network
```

This gives a **stable Docker service name** (`gpu-server:8082`), zero network overhead (same bridge network), automatic restart handling via Docker Compose, and no scripts to maintain. The RTX 5060 Ti would be accessed via GPU passthrough while the V100 remains on Windows for `llama-server.exe`.

---

## Risk assessment and hidden pitfalls

**For the `netsh portproxy` approach (primary recommendation):**

- **WSL2 IP instability** is the biggest operational risk. The IP changes on every WSL restart or Windows reboot. The Task Scheduler script mitigates this, but there is a race condition at boot: WSL may not be ready when the script runs. The retry loop in the script handles this, but edge cases exist. For higher reliability, consider the community project `cjnuk/wsl2-port-mapper`, a Go-based Windows service that continuously monitors and reconciles portproxy rules.
- **Port 8080 must NOT get a portproxy rule.** Creating a portproxy on `0.0.0.0:8080` would conflict with `llama-server.exe` which already binds that port natively on Windows. The script should only forward port 8082.
- **Windows Firewall and Hyper-V Firewall are separate.** On Windows 11 22H2+, the Hyper-V firewall (distinct from Windows Defender Firewall) can silently block traffic. The `Set-NetFirewallHyperVVMSetting` command is critical.
- **VPN impact is minimal.** `netsh portproxy` operates at the Windows network layer and generally coexists with VPNs. However, VPN clients that modify routing tables (e.g., Cisco AnyConnect in tunnel-all mode) can occasionally disrupt the `vEthernet (WSL)` adapter. Test with your VPN active.
- **Security exposure**: The portproxy binds to `0.0.0.0`, making port 8082 accessible from the LAN. If this is undesirable, restrict the firewall rule to `LocalAddress 127.0.0.1` or the Docker network range, or change `listenaddress` to `127.0.0.1` (though this may not work with vpnkit's connection path — test first).

**For mirrored networking (avoid for now):** TCP stalls in Docker containers (moby/moby#48201), Docker Desktop startup failures requiring `nestedVirtualization=true` (docker/for-win#14691), and broken port publishing (microsoft/WSL#10494) make this approach unreliable as of early 2026. Monitor these GitHub issues — once resolved, mirrored mode will likely become the recommended approach.

---

## Conclusion

The Docker ↔ WSL2 networking gap exists because Docker Desktop and WSL2 user distros occupy **isolated Hyper-V virtual networks** with no default routing between them. `host.docker.internal` routes through vpnkit to the Windows host, but WSL2's automatic localhost forwarding is too unreliable to bridge the final hop. **`netsh interface portproxy`** solves this by creating a real Windows-level TCP listener that makes port 8082 deterministically reachable from Docker containers — it requires a single PowerShell script running at logon to handle WSL2's dynamic IP, and it leaves the existing `llama-server.exe` on port 8080 completely untouched.

The relay's 404 errors on `/rerank` and `/colbert-encode` are a separate, application-level issue — verify those routes exist in `gpu_server.py` by curling directly from inside WSL2 before blaming the network layer. Once the portproxy is in place and the endpoints are confirmed working, the full retrieval pipeline (dense + sparse + reranker + ColBERT) should function through a single stable address: `host.docker.internal:8082`.