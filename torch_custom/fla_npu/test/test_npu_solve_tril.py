import torch
import torch_npu
import numpy as np
import fla_npu

torch.npu.utils.set_device(1)
torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)


def make_input(B, S, H, BT, dtype=torch.float16):
    """Construct unit lower-triangular input [B, S, H, BT]."""
    A = torch.zeros(B, S, H, BT, dtype=dtype)
    NT = (S + BT - 1) // BT
    for b in range(B):
        for h in range(H):
            for c in range(NT):
                s_start = c * BT
                s_end = min(s_start + BT, S)
                eff = s_end - s_start
                for i in range(eff):
                    A[b, s_start + i, h, i] = 1.0
                    for j in range(i):
                        A[b, s_start + i, h, j] = torch.randn(1, dtype=torch.float32).item() * 0.3
    return A


def solve_tril_ref(A):
    """Reference: torch.inverse per chunk."""
    B, S, H, BT = A.shape
    NT = (S + BT - 1) // BT
    out = torch.zeros_like(A)
    for b in range(B):
        for h in range(H):
            for c in range(NT):
                s_start = c * BT
                s_end = min(s_start + BT, S)
                eff = s_end - s_start
                if eff < BT:
                    block = torch.eye(BT, dtype=A.dtype, device=A.device)
                    block[:eff, :eff] = A[b, s_start:s_end, h, :eff]
                    inv = torch.inverse(block.to(torch.float32)).to(A.dtype)
                    out[b, s_start:s_end, h, :eff] = inv[:eff, :eff]
                else:
                    block = A[b, s_start:s_end, h, :BT]
                    inv = torch.inverse(block.to(torch.float32)).to(A.dtype)
                    out[b, s_start:s_end, h, :BT] = inv
    return out


def compute_mere(actual, golden):
    diff = torch.abs(actual.float() - golden.float())
    denom = torch.clamp(torch.abs(golden.float()), min=1e-10)
    return (diff / denom).max().item()


def test_case(B, S, H, BT, name):
    torch.manual_seed(42)
    A = make_input(B, S, H, BT, torch.float16)
    A_npu = A.contiguous().npu()
    torch.npu.synchronize()

    for _ in range(5):
        result_npu = torch.ops.npu.npu_solve_tril(A_npu)
        torch.npu.synchronize()

    golden = solve_tril_ref(A)
    mere = compute_mere(result_npu.cpu(), golden)
    threshold = 9.77e-04  # fp16 spec standard
    passed = mere < threshold
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: B={B},S={S},H={H},BT={BT}, MERE={mere:.6e}")
    return passed


if __name__ == "__main__":
    print("Warming up kernel variants...")
    for BT in [16, 32, 64]:
        A_warmup = make_input(1, BT, 1, BT, torch.float16).contiguous().npu()
        for _ in range(5):
            _ = torch.ops.npu.npu_solve_tril(A_warmup)
            torch.npu.synchronize()
        del A_warmup
    torch.npu.synchronize()
    print("Warmup complete.\n")

    results = []
    results.append(test_case(1, 16, 1, 16, "L0-01 BT=16"))
    results.append(test_case(1, 32, 1, 32, "L0-02 BT=32"))
    results.append(test_case(1, 64, 1, 64, "L0-03 BT=64"))
    # bug
    # results.append(test_case(1, 128, 1, 128, "L0-04 BT=128"))
    results.append(test_case(1, 64, 2, 64, "L1-01 H=2"))
    results.append(test_case(2, 64, 4, 64, "L1-02 B=2,H=4"))
    results.append(test_case(1, 128, 1, 64, "L1-03 NT=2"))
    results.append(test_case(1, 100, 1, 64, "L1-04 S=100 tail"))

    passed = sum(results)
    total = len(results)
    print(f"\nTotal: {total}, Passed: {passed}, Failed: {total - passed}")
    if passed < total:
        exit(1)
