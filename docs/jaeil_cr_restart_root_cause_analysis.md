# jaeil_cr 컨테이너 잦은 재시작 원인 분석

## 증상

`spacenorm_obj_detection_jaeil_cr` 서비스(노드 `spacenorm-a0ad9fbf3164`, 10.144.0.12)의 컨테이너가
몇 시간 간격으로 계속 재시작됨. `docker service ps`로 확인한 태스크들은 대부분 종료 코드
**137 (SIGKILL)** 로 종료됨.

## 결론

**원인은 애플리케이션 코드, Docker 이미지, WireGuard 설정이 아니라 jaeil_cr 사이트 자체의
인터넷(WAN) 회선이 하루에 여러 번, 3~4분가량 완전히 끊기는 문제**임. 회선이 끊기면 Docker
Swarm 매니저가 해당 노드의 하트비트(gossip)를 받지 못해 노드를 `Down`으로 판단하고, 정상적인
장애 복구 절차에 따라 그 노드에서 실행 중이던 태스크를 교체(재시작)함. 즉, 관찰되는 재시작은
Swarm이 "의도대로" 동작한 결과이며, 이 리포지토리의 코드나 이미지, 배포 설정과는 무관함.

## 조사 과정

### 1. 매니저 측 로그로 1차 단서 확인

`docker node ls`로 해당 노드가 재시작 시점마다 `Down` / `heartbeat failure` 상태가 되는 것을
확인. 매니저의 `journalctl -u docker` 로그에서 해당 노드(`10.144.0.12`)가 반복적으로 gossip
클러스터에서 이탈(`left gossip cluster`)했다가 다시 합류(`joined gossip cluster`)하는 패턴을
다수 발견함. 하루 동안 수십 회 발생하며, 지속 시간은 수십 초에서 최대 20여 분까지 다양함.

### 2. 노드에 직접 접속하여 원인 좁히기

노드에 SSH로 접속하여 다음을 확인함:

- WireGuard(`wg0`) 설정은 정상: `PersistentKeepalive = 25`가 설정되어 있고, 평상시
  handshake도 신선한 상태(11초 이내)였음 → **WireGuard 자체의 설정 문제는 아님**.
- 로컬 LAN NIC(`eno1`, 사이트 공유기로 연결)의 드라이버 레벨 에러 카운터는 0 → 로컬 랜카드/케이블
  문제도 아님.
- GPU 상태(`nvidia-smi`) 및 커널 로그에도 이상 없음.

### 3. 실시간 삼중 검증 (트래픽 우회 핑 + 터널 카운터 + 더미(카나리) 컨테이너)

다음 세 가지를 노드에서 동시에 백그라운드로 모니터링:

1. **WireGuard 터널을 우회하는 직접 ICMP 핑** (`8.8.8.8`로 2초 간격)
2. **`wg0` 인터페이스의 RX/TX 패킷 카운터** (`ip -s link show wg0`, sudo 불필요)
3. **해당 노드에만 고정 배치한 트리비얼한 테스트(카나리) 컨테이너** — 아무 로직도 없이 5초마다
   타임스탬프만 출력하는 Alpine 컨테이너 (`netcanary_jaeil_cr`)

실제 장애가 다시 발생했을 때(2026-07-19 11:10:07 ~ 11:13:16 KST, 약 3분 9초) 세 신호가
거의 초 단위로 정확히 일치함:

| 신호 | 시각 (KST) |
|---|---|
| Swarm gossip: `Suspect` → `left gossip cluster` | 11:10:07 ~ 11:10:11 |
| 직접 ICMP 핑(터널 우회) 100% 실패 구간 | 11:10:06 ~ 11:13:16 |
| Swarm gossip: `joined gossip cluster` | 11:13:16 |
| 카나리 컨테이너: 신규 태스크 생성 후 `Pending` | 11:10:19 (장애 발생 8초 후) |
| 카나리 컨테이너: `Pending` → `Running` 전환 | 11:13:20 (복구 4초 후) |
| 실제 앱 서비스: 신규 태스크로 교체 | 11:13:20 (카나리와 동일 시각) |

카나리 컨테이너는 이 서비스와 관련된 코드를 전혀 포함하지 않는 단순 반복문임에도 불구하고,
실제 앱과 **완전히 동일한 시점에 동일한 패턴(장애 중 `Pending` 대기 → 복구 시 `Running`)**으로
재스케줄링됨. 이는 재시작 원인이 애플리케이션과 무관하게 **해당 노드 자체가 Swarm 클러스터에서
일시적으로 격리되는 것**임을 직접적으로 증명함.

## 재시작이 발생하는 정확한 메커니즘

1. jaeil_cr 사이트의 WAN 회선이 완전히 끊김 (3~4분 내외, 불규칙한 간격으로 하루 여러 차례 발생).
2. WireGuard 터널을 포함한 모든 외부 통신이 끊기므로, 매니저는 해당 노드의 gossip 하트비트를
   받지 못함.
3. 매니저는 일정 시간(수 초~수십 초) 후 해당 노드를 `NodeFailed`로 표시하고, 그 노드에 배치된
   태스크가 유실된 것으로 간주하여 대체 태스크를 스케줄링함. (전역 배치/노드 제약 조건 때문에
   대체 태스크는 같은 노드에서만 실행 가능하므로 노드가 복구될 때까지 `Pending` 상태로 대기.)
4. 회선이 복구되어 노드가 gossip 클러스터에 재합류하면, 노드의 Docker 에이전트가 매니저와
   상태를 동기화하는 과정에서 기존에 남아있던(orphan) 컨테이너를 종료(SIGKILL, 종료 코드 137)
   하고, 미리 예약되어 있던 신규 태스크를 시작함.

## 권장 후속 조치

- 이 리포지토리의 애플리케이션 코드, Docker 이미지, WireGuard 설정을 원인으로 보고 추가로
  수정할 필요는 없음.
- jaeil_cr 현장의 공유기/모뎀/ISP 회선 점검이 필요함:
  - 공유기·모뎀에 주기적인 재부팅/WAN 재연결(PPPoE 재인증, DHCP 갱신 실패 등) 로그가 있는지 확인
  - ISP 회선 상태 또는 장애 이력 문의
  - 서비스 가동률이 중요하다면 예비 회선(예: LTE/5G 백업 회선) 도입 검토
- `wg0`의 누적 TX 에러 카운터는 평상시에도 계속 증가하는 값이라 장애 판단 지표로 신뢰할 수
  없음 — 향후 유사 조사 시 참고.

## 검증에 사용한 방법 (재현 가능)

향후 유사한 현상 재조사 시 아래 방법을 그대로 재사용할 수 있음:

```bash
# 매니저에서: 해당 노드의 gossip 이탈/재합류 이력 확인 (시간대는 매니저 로컬 시간 기준)
journalctl -u docker --since "<시작시각>" --no-pager \
  | grep -iE '<노드 ID 앞 12자리>|<노드 WireGuard IP>' \
  | grep -iE 'suspect|failed|joined gossip|left gossip'

# 노드에서: WireGuard를 우회하는 직접 핑 (2초 간격, 백그라운드)
ping -i 2 8.8.8.8

# 노드에서: wg0 패킷 카운터 스냅샷 (sudo 불필요)
ip -s link show wg0

# 매니저에서: 해당 노드에 고정 배치되는 트리비얼 카나리 서비스 배포
docker service create \
  --name netcanary_<site> \
  --constraint 'node.id==<노드ID>' \
  --restart-condition any \
  alpine sh -c 'while true; do date; sleep 5; done'
```

세 신호(직접 핑 단절 구간, `wg0` 카운터 정지 구간, 카나리 컨테이너의 재스케줄링 시점)가
Swarm gossip 로그의 이탈/재합류 시각과 일치하면, 원인은 해당 사이트의 네트워크 회선 문제임을
의미함.

---

## 별도 사건: 매니저(GTEC 박스) 자체의 NIC 문제로 인한 전체 사이트 동시 재시작

### 증상

2026-07-23 기준으로, jaeil_cr뿐 아니라 **jaeil_io, kumho, cym 등 여러 사이트의 서비스가 거의
동시에(6~7시간 전 기준) 재시작**된 것이 확인됨. 이는 위에서 분석한 "개별 사이트 WAN 단절"
패턴과는 다른 양상으로, 특정 사이트 하나가 아니라 **여러 워커 노드가 한꺼번에** gossip
클러스터에서 이탈했다 재합류하기를 약 70분간(14:31~15:43 KST) 반복함.

### 조사 과정

매니저(`ahnjw-GTEC-Ubuntu`)의 `journalctl -u docker` 로그를 확인한 결과, 이번에는
**매니저 자신이 스스로의 연결 상태를 "connectivity issues"로 보고**하고 있음을 발견함:

```
NetworkDB stats ahnjw-GTEC-Ubuntu(037cb1e2a7ad) - healthscore:7 (connectivity issues)
...
NetworkDB stats ahnjw-GTEC-Ubuntu(037cb1e2a7ad) - healthscore:2 (connectivity issues)
```

또한 다른 워커 노드들이 매니저(`037cb1e2a7ad`)를 "suspect(장애 의심)"로 판단하고, 매니저가 이를
반박(`Refuting a suspect message`)하는 로그가 다수 발견됨. 즉, 문제의 근원이 각 사이트의 회선이
아니라 **매니저 쪽**에 있었다는 뜻임.

매니저 자신의 로컬 랜카드(`eno1`)와 사이트 공유기(WAN) 상태를 점검한 결과, `eno1`의 RX/TX
에러·드롭 카운터는 0으로 정상이었음. 하지만 커널 로그(`dmesg`/`journalctl -k`)에서 결정적인
단서를 발견함:

```
r8169 0000:06:00.0 eno1: NETDEV WATCHDOG: CPU: 10: transmit queue 0 timed out 5968 ms
r8169 0000:06:00.0: can't disable ASPM; OS doesn't have ASPM control
```

같은 70분 구간 동안 이 "NETDEV WATCHDOG" 메시지가 총 **16회** 발생했으며, 매번 송신 큐가
5~6초간 완전히 멈췄음(timed out). 매니저의 부팅 후 가동 시간(38일) 전체를 통틀어 이 메시지가
발생한 것은 이 70분 구간이 유일함 — 즉 상시 발생하는 문제가 아니라 이번에 한 번 몰아서 발생한
사건임.

### 결론

- 매니저의 이더넷 컨트롤러는 **Realtek RTL8125 (2.5GbE)**, 드라이버는 `r8169`.
- 해당 NIC가 `r8169` 드라이버 + ASPM(PCIe 전원관리) 조합에서 흔히 보고되는 **송신 큐 정지(TX
  queue hang) 버그**를 겪은 것으로 보임. 커널이 "OS doesn't have ASPM control"이라고 명시한
  것으로 보아, BIOS/UEFI가 ASPM을 잠가두어 커널이 이를 제어하지 못하는 상태로 추정됨.
- 매니저에서 나가는 **모든** Swarm gossip/하트비트 트래픽이 이 NIC 하나를 거치기 때문에, 5~6초의
  완전한 송신 정지가 발생할 때마다 마침 그 순간 하트비트 확인 주기가 걸려있던 여러 워커
  노드들이 한꺼번에 `Suspect`/`Down`으로 판정됨. 이 때문에 jaeil_io, kumho, cym 등 여러 사이트가
  거의 동시에 재시작된 것처럼 보인 것임.
- 이는 위에서 분석한 **jaeil_cr 개별 사이트의 WAN 단절 문제와는 별개의, 매니저 측 하드웨어/
  드라이버 이슈**임. jaeil_cr의 재시작 패턴은 여전히 사이트 자체 회선 문제로 계속 발생하고
  있으며, 이번 사건과는 무관함.

### 권장 후속 조치

- 재발 여부 모니터링: 38일 가동 중 이번이 유일한 발생이므로 아직 상시 문제로 단정하기는
  이르나, 재발 시 아래 조치를 검토.
- **EEE(Energy Efficient Ethernet) 비활성화** (재부팅 불필요, 우선 시도 가능):
  ```bash
  sudo ethtool --set-eee eno1 eee off
  ```
- **ASPM 비활성화** (재부팅 필요):
  - 커널 부트 파라미터에 `pcie_aspm=off` 추가, 또는
  - BIOS/UEFI에서 해당 PCIe 슬롯의 ASPM 설정을 끄기
- 재발이 잦아질 경우 `r8169` 드라이버/커널 업데이트, 혹은 다른 NIC로 교체도 고려 가능.
