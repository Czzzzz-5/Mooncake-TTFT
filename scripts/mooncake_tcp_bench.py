#!/usr/bin/env python
"""Mooncake transfer engine TCP 带宽 bench（瓶颈排查第 1 步：拆包实验）。

server: 起 engine，注册 buffer（CPU managed 或 GPU torch tensor），把 "ip:rpc_port" 写到 --addr-file
client: 对 server 做批量同步传输（push=sync_write / pull=sync_read），按 --sizes-mb 扫描传输尺寸，
        每个尺寸重复 --iters 次，输出 avg/median/min/max 带宽。

用法（详见 MOONCAKE_TCP_BANDWIDTH_PLAN.md §3 第 1 步）：

  # 终端 1：起 server（cpu / gpu 各一次）
  python mooncake_tcp_bench.py server --buf cpu --buf-gb 2
  CUDA_VISIBLE_DEVICES=<空闲卡> python mooncake_tcp_bench.py server --buf gpu --buf-gb 2

  # 终端 2：client 扫描尺寸（方向 push / pull 各跑一次）
  python mooncake_tcp_bench.py client --buf cpu --dir push --sizes-mb 115,512,1024 --iters 8
  CUDA_VISIBLE_DEVICES=<空闲卡> python mooncake_tcp_bench.py client --buf gpu --dir push \
      --sizes-mb 115,512,1024 --iters 8
  # 复刻 vLLM 真实形状：115 MB 拆成 1850 个 ~64KB 描述符
  CUDA_VISIBLE_DEVICES=<空闲卡> python mooncake_tcp_bench.py client --buf gpu --dir push \
      --sizes-mb 115 --descs 1850 --iters 8
"""
import argparse
import os
import time

ADDR_FILE_DEFAULT = "/tmp/mooncake_bench_server.txt"


def make_engine():
    from mooncake.engine import TransferEngine

    # 长传输描述符多，直接放开上限（9.6 实验踩过的坑）
    os.environ.setdefault("MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER", "65535")
    os.environ.setdefault("MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER", "65535")
    e = TransferEngine()
    ret = e.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "")
    assert ret == 0, f"engine init failed: {ret}"
    return e


def make_buffer(e, buf_kind, size):
    """分配并注册 buffer，返回 (基地址, 底层对象hold住防GC)。"""
    if buf_kind == "cpu":
        addr = e.allocate_managed_buffer(size)
        e.write_bytes_to_buffer(addr, bytes((i * 7 + 3) & 0xFF for i in range(4096)), 4096)
        return addr, None
    import torch

    t = torch.empty(size, dtype=torch.uint8, device="cuda")
    t[:4096] = torch.tensor([(i * 7 + 3) & 0xFF for i in range(4096)],
                            dtype=torch.uint8, device="cuda")
    ret = e.register_memory(t.data_ptr(), size)
    assert ret == 0, f"register_memory failed: {ret}"
    return t.data_ptr(), t


def read_local(e, buf_kind, addr, n, hold):
    if buf_kind == "cpu":
        return e.read_bytes_from_buffer(addr, n)
    import torch

    return bytes(hold[:n].cpu().numpy())


def run_server(buf_kind, buf_gb, addr_file):
    e = make_engine()
    size = int(buf_gb * (1 << 30))
    addr, hold = make_buffer(e, buf_kind, size)
    if os.path.exists(addr_file):
        os.remove(addr_file)
    name = "127.0.0.1:" + str(e.get_rpc_port())
    with open(addr_file, "w") as f:
        f.write(name)
    print(f"[server] {name} buf={buf_kind} size={size / 1e9:.2f} GB addr=0x{addr:x}",
          flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def run_client(buf_kind, direction, sizes_mb, descs, iters, addr_file):
    import statistics

    e = make_engine()
    for _ in range(100):
        if os.path.exists(addr_file):
            break
        time.sleep(0.2)
    target = open(addr_file).read().strip()
    remote_base = e.get_first_buffer_address(target)
    max_size = int(max(sizes_mb) * 1e6)
    local, hold = make_buffer(e, buf_kind, max_size)
    print(f"[client] target={target} remote_base=0x{remote_base:x} local=0x{local:x} "
          f"buf={buf_kind} dir={direction}", flush=True)

    def do_transfer(size):
        chunk = size // descs
        lens = [chunk] * (descs - 1) + [size - chunk * (descs - 1)]
        local_addrs = [local + i * chunk for i in range(descs)]
        remote_addrs = [remote_base + i * chunk for i in range(descs)]
        if direction == "push":  # 客户端本地 → 服务端（复刻 prefill 侧 sync_write）
            return e.batch_transfer_sync_write(target, local_addrs, remote_addrs, lens)
        # pull：服务端 → 客户端本地（复刻 decode 侧 sync_read）
        return e.batch_transfer_sync_read(target, local_addrs, remote_addrs, lens)

    for size_mb in sizes_mb:
        size = int(size_mb * 1e6)
        warm_ret = do_transfer(size)  # 预热，不计入统计
        times = []
        for _ in range(iters):
            if buf_kind == "gpu":
                import torch

                torch.cuda.synchronize()  # 排空前序 CUDA 操作，避免计时串味
            t0 = time.perf_counter()
            ret = do_transfer(size)
            times.append(time.perf_counter() - t0)
            assert ret == 0, f"transfer failed: ret={ret}"
        if direction == "pull" and warm_ret == 0:
            expect = bytes((i * 7 + 3) & 0xFF for i in range(16))
            got = read_local(e, buf_kind, local, 16, hold)
            assert got == expect, f"data mismatch: {got.hex()} != {expect.hex()}"
        mbps = [size / t / 1e6 for t in times]
        ms = [t * 1e3 for t in times]
        print(f"[client] buf={buf_kind} dir={direction} size={size_mb:g} MB descs={descs} "
              f"| avg={statistics.mean(mbps):7.0f} MB/s ({statistics.mean(ms):6.1f} ms) "
              f"| median={statistics.median(mbps):7.0f} | min={min(mbps):7.0f} "
              f"| max={max(mbps):7.0f} | n={iters}", flush=True)
    print("[client] done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["server", "client"])
    ap.add_argument("--buf", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--buf-gb", type=float, default=2.0, help="server buffer 大小(GB)")
    ap.add_argument("--dir", choices=["push", "pull"], default="push")
    ap.add_argument("--sizes-mb", default="115,512,1024", help="逗号分隔的传输尺寸")
    ap.add_argument("--descs", type=int, default=1, help="每次传输拆成的描述符数")
    ap.add_argument("--iters", type=int, default=8, help="每个尺寸的重复次数")
    ap.add_argument("--addr-file", default=ADDR_FILE_DEFAULT)
    args = ap.parse_args()

    if args.mode == "server":
        run_server(args.buf, args.buf_gb, args.addr_file)
    else:
        sizes = [float(x) for x in args.sizes_mb.split(",")]
        run_client(args.buf, args.dir, sizes, args.descs, args.iters, args.addr_file)


if __name__ == "__main__":
    main()
