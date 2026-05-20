// =============================================================================
//  cub_sort_gwf.cuh
//
//  Drop-in CUB replacements for:
//    · fast_parallel_sort()   (sorts uint64_t key/val pairs, used for intv)
//    · sort_gwf_diags()       (sorts gwf_diag_t by .vd field)
//
//  WHY A DISPATCH INSTEAD OF A SINGLE INSTANTIATION
//  ─────────────────────────────────────────────────
//  cub::BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD> requires
//  ITEMS_PER_THREAD (IPT) as a compile-time constant because each thread
//  holds its IPT items in *registers* — the compiler must know the array
//  size at translation time to allocate register slots.
//
//  Since n is variable at runtime, we instantiate five template
//  specialisations (IPT ∈ {1, 2, 4, 8, 16}) covering up to
//  32 × 16 = 512 elements, then dispatch with a runtime branch:
//
//    n ≤  32  →  IPT = 1
//    n ≤  64  →  IPT = 2
//    n ≤ 128  →  IPT = 4
//    n ≤ 256  →  IPT = 8
//    n ≤ 512  →  IPT = 16
//
//  Elements that do not exist are padded with UINT64_MAX (sentinel), which
//  sorts to the tail and is never written back because the write loop guards
//  on (idx < n).
//
//  SHARED MEMORY: UNION TRICK
//  ──────────────────────────
//  Each specialisation needs its own TempStorage type. If we declared five
//  separate __shared__ variables the compiler would allocate their sizes
//  *cumulatively*, wasting shared memory and reducing occupancy.
//  Instead we wrap them all in a __shared__ union:
//
//    __shared__ union { TempStorage1 ts1; TempStorage2 ts2; ... } u;
//
//  The compiler allocates max(sizeof(ts1), ..., sizeof(ts16)) bytes, and
//  every active branch reuses the same physical shared memory region.
//  Only one branch is active per kernel invocation, so there is no aliasing
//  hazard.
//
//  HOW CUB BLOCKED ARRANGEMENT WORKS
//  ───────────────────────────────────
//  Thread tid owns the IPT register slots that correspond to global indices
//  [tid*IPT, (tid+1)*IPT − 1].  After Sort(), the output is still in
//  blocked arrangement but globally ascending: thread 0 holds the smallest
//  IPT keys, thread 1 the next IPT, etc.  Writing back to the same index
//  range therefore produces a sorted array in global memory.  ✓
//
//  USAGE
//  ─────
//  Replace the two calls in gwfa.cu:
//
//    fast_parallel_sort(buf->tmp.vd0, buf->tmp.vd1, cnt.n_tmp)
//    → fast_parallel_sort_cub(buf->tmp.vd0, buf->tmp.vd1, cnt.n_tmp)
//
//    sort_gwf_diags(c, n_c)
//    → sort_gwf_diags_cub(c, n_c)
//
//  Also #include this header at the top of gwfa.cu.
// =============================================================================

#pragma once
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>
#include "gwfa.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Section 1 — fast_parallel_sort_cub
//  Sorts n pairs (keys[], vals[]) of type uint64_t in ascending key order.
//  Direct replacement for fast_parallel_sort().
// ─────────────────────────────────────────────────────────────────────────────

// Convenience aliases for the five BlockRadixSort instantiations.
// Key = uint64_t, Value = uint64_t, 32 threads, varying IPT.
typedef cub::BlockRadixSort<uint64_t, 32,  1, uint64_t> BRS_u64_ipt1;
typedef cub::BlockRadixSort<uint64_t, 32,  2, uint64_t> BRS_u64_ipt2;
typedef cub::BlockRadixSort<uint64_t, 32,  4, uint64_t> BRS_u64_ipt4;
typedef cub::BlockRadixSort<uint64_t, 32,  8, uint64_t> BRS_u64_ipt8;
typedef cub::BlockRadixSort<uint64_t, 32, 16, uint64_t> BRS_u64_ipt16;

// Union that holds the TempStorage for whichever IPT branch is active.
// Physical size = max(sizeof(ts1), ..., sizeof(ts16)).
struct __align__(128) SortTempStorage_u64 {
    union {
        typename BRS_u64_ipt1 ::TempStorage ts1;
        typename BRS_u64_ipt2 ::TempStorage ts2;
        typename BRS_u64_ipt4 ::TempStorage ts4;
        typename BRS_u64_ipt8 ::TempStorage ts8;
        typename BRS_u64_ipt16::TempStorage ts16;
    };
};

// ── IPT = 1 (n ≤ 32) ─────────────────────────────────────────────────────────
__device__ __forceinline__
static void _sort_u64_ipt1(uint64_t *keys, uint64_t *vals, int n,
                             SortTempStorage_u64 &ts)
{
    const int tid = threadIdx.x;
    uint64_t k[1], v[1];

    int idx = tid;                               // IPT=1: thread tid → index tid
    k[0] = (idx < n) ? keys[idx] : UINT64_MAX;
    v[0] = (idx < n) ? vals[idx] : 0ULL;

    BRS_u64_ipt1(ts.ts1).Sort(k, v);
    __syncthreads();

    if (idx < n) { keys[idx] = k[0]; vals[idx] = v[0]; }
    __syncthreads();
}

// ── IPT = 2 (n ≤ 64) ─────────────────────────────────────────────────────────
__device__ __forceinline__
static void _sort_u64_ipt2(uint64_t *keys, uint64_t *vals, int n,
                             SortTempStorage_u64 &ts)
{
    const int tid = threadIdx.x;
    uint64_t k[2], v[2];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = tid * 2 + i;
        k[i] = (idx < n) ? keys[idx] : UINT64_MAX;
        v[i] = (idx < n) ? vals[idx] : 0ULL;
    }

    BRS_u64_ipt2(ts.ts2).Sort(k, v);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = tid * 2 + i;
        if (idx < n) { keys[idx] = k[i]; vals[idx] = v[i]; }
    }
    __syncthreads();
}

// ── IPT = 4 (n ≤ 128) ────────────────────────────────────────────────────────
__device__ __forceinline__
static void _sort_u64_ipt4(uint64_t *keys, uint64_t *vals, int n,
                             SortTempStorage_u64 &ts)
{
    const int tid = threadIdx.x;
    uint64_t k[4], v[4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid * 4 + i;
        k[i] = (idx < n) ? keys[idx] : UINT64_MAX;
        v[i] = (idx < n) ? vals[idx] : 0ULL;
    }

    BRS_u64_ipt4(ts.ts4).Sort(k, v);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid * 4 + i;
        if (idx < n) { keys[idx] = k[i]; vals[idx] = v[i]; }
    }
    __syncthreads();
}

// ── IPT = 8 (n ≤ 256) ────────────────────────────────────────────────────────
__device__ __forceinline__
static void _sort_u64_ipt8(uint64_t *keys, uint64_t *vals, int n,
                             SortTempStorage_u64 &ts)
{
    const int tid = threadIdx.x;
    uint64_t k[8], v[8];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid * 8 + i;
        k[i] = (idx < n) ? keys[idx] : UINT64_MAX;
        v[i] = (idx < n) ? vals[idx] : 0ULL;
    }

    BRS_u64_ipt8(ts.ts8).Sort(k, v);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid * 8 + i;
        if (idx < n) { keys[idx] = k[i]; vals[idx] = v[i]; }
    }
    __syncthreads();
}

// ── IPT = 16 (n ≤ 512) ───────────────────────────────────────────────────────
__device__ __forceinline__
static void _sort_u64_ipt16(uint64_t *keys, uint64_t *vals, int n,
                              SortTempStorage_u64 &ts)
{
    const int tid = threadIdx.x;
    uint64_t k[16], v[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx = tid * 16 + i;
        k[i] = (idx < n) ? keys[idx] : UINT64_MAX;
        v[i] = (idx < n) ? vals[idx] : 0ULL;
    }

    BRS_u64_ipt16(ts.ts16).Sort(k, v);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx = tid * 16 + i;
        if (idx < n) { keys[idx] = k[i]; vals[idx] = v[i]; }
    }
    __syncthreads();
}

// ── Public entry point ────────────────────────────────────────────────────────
// Replaces fast_parallel_sort(keys, vals, n) in gwfa.cu.
// Shared memory for TempStorage is allocated once (the union); the compiler
// reserves max(sizeof(ts1),..,sizeof(ts16)) bytes regardless of which branch
// is taken at runtime.
__device__ __inline__
void fast_parallel_sort_cub(uint64_t *keys, uint64_t *vals, int n)
{
    __shared__ SortTempStorage_u64 _ts;         // one allocation, covers all IPT

    if      (n <=  32) _sort_u64_ipt1 (keys, vals, n, _ts);
    else if (n <=  64) _sort_u64_ipt2 (keys, vals, n, _ts);
    else if (n <= 128) _sort_u64_ipt4 (keys, vals, n, _ts);
    else if (n <= 256) _sort_u64_ipt8 (keys, vals, n, _ts);
    else               _sort_u64_ipt16(keys, vals, n, _ts);
    // n > 512: not expected given max_a sizing; add an IPT=32 branch if needed.
}


// =============================================================================
//  Section 2 — sort_gwf_diags_cub
//  Sorts an array of gwf_diag_t in ascending order of the .vd field.
//  Direct replacement for sort_gwf_diags(gwf_diag_t *c, int32_t n).
//
//  Strategy: pack .k (int32_t) and .xo (uint32_t) into a single uint64_t
//  value so we can use CUB's key-value sort with uint64_t keys (vd) and
//  uint64_t values (packed k|xo).  After sorting, unpack back into the
//  struct array.
//
//  Packing:
//    packed_val = ((uint64_t)(uint32_t)k << 32) | (uint64_t)xo
//  Unpacking:
//    k   = (int32_t)(packed_val >> 32)     // bit-cast back; preserves sign
//    xo  = (uint32_t)(packed_val & 0xFFFFFFFFULL)
// =============================================================================

// Aliases for the gwf_diag sort (same key/val types as above, so we reuse
// the same TempStorage union — only ONE shared union is needed per kernel).

__device__ __forceinline__
static uint64_t _pack_diag(int32_t k, uint32_t xo)
{
    return ((uint64_t)(uint32_t)k << 32) | (uint64_t)xo;
}

__device__ __forceinline__
static void _unpack_diag(uint64_t packed, int32_t &k, uint32_t &xo)
{
    k  = (int32_t)(packed >> 32);
    xo = (uint32_t)(packed & 0xFFFFFFFFULL);
}

// ── IPT = 1 ──────────────────────────────────────────────────────────────────
__device__ __forceinline__
static void _sort_diag_ipt1(gwf_diag_t *c, int32_t n,
                              SortTempStorage_u64 &ts)
{
    const int tid = threadIdx.x;
    uint64_t k[1], v[1];

    int idx = tid;
    k[0] = (idx < n) ? c[idx].vd              : UINT64_MAX;
    v[0] = (idx < n) ? _pack_diag(c[idx].k, c[idx].xo) : 0ULL;

    BRS_u64_ipt1(ts.ts1).Sort(k, v);
    __syncthreads();

    if (idx < n) {
        c[idx].vd = k[0];
        _unpack_diag(v[0], c[idx].k, c[idx].xo);
    }
    __syncthreads();
}

// ── IPT = 2 ──────────────────────────────────────────────────────────────────
__device__ __forceinline__
static void _sort_diag_ipt2(gwf_diag_t *c, int32_t n,
                              SortTempStorage_u64 &ts)
{
    const int tid = threadIdx.x;
    uint64_t k[2], v[2];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = tid * 2 + i;
        k[i] = (idx < n) ? c[idx].vd                    : UINT64_MAX;
        v[i] = (idx < n) ? _pack_diag(c[idx].k, c[idx].xo) : 0ULL;
    }

    BRS_u64_ipt2(ts.ts2).Sort(k, v);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int idx = tid * 2 + i;
        if (idx < n) {
            c[idx].vd = k[i];
            _unpack_diag(v[i], c[idx].k, c[idx].xo);
        }
    }
    __syncthreads();
}

// ── IPT = 4 ──────────────────────────────────────────────────────────────────
__device__ __forceinline__
static void _sort_diag_ipt4(gwf_diag_t *c, int32_t n,
                              SortTempStorage_u64 &ts)
{
    const int tid = threadIdx.x;
    uint64_t k[4], v[4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid * 4 + i;
        k[i] = (idx < n) ? c[idx].vd                    : UINT64_MAX;
        v[i] = (idx < n) ? _pack_diag(c[idx].k, c[idx].xo) : 0ULL;
    }

    BRS_u64_ipt4(ts.ts4).Sort(k, v);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid * 4 + i;
        if (idx < n) {
            c[idx].vd = k[i];
            _unpack_diag(v[i], c[idx].k, c[idx].xo);
        }
    }
    __syncthreads();
}

// ── IPT = 8 ──────────────────────────────────────────────────────────────────
__device__ __forceinline__
static void _sort_diag_ipt8(gwf_diag_t *c, int32_t n,
                              SortTempStorage_u64 &ts)
{
    const int tid = threadIdx.x;
    uint64_t k[8], v[8];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid * 8 + i;
        k[i] = (idx < n) ? c[idx].vd                    : UINT64_MAX;
        v[i] = (idx < n) ? _pack_diag(c[idx].k, c[idx].xo) : 0ULL;
    }

    BRS_u64_ipt8(ts.ts8).Sort(k, v);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = tid * 8 + i;
        if (idx < n) {
            c[idx].vd = k[i];
            _unpack_diag(v[i], c[idx].k, c[idx].xo);
        }
    }
    __syncthreads();
}

// ── IPT = 16 ─────────────────────────────────────────────────────────────────
__device__ __forceinline__
static void _sort_diag_ipt16(gwf_diag_t *c, int32_t n,
                               SortTempStorage_u64 &ts)
{
    const int tid = threadIdx.x;
    uint64_t k[16], v[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx = tid * 16 + i;
        k[i] = (idx < n) ? c[idx].vd                    : UINT64_MAX;
        v[i] = (idx < n) ? _pack_diag(c[idx].k, c[idx].xo) : 0ULL;
    }

    BRS_u64_ipt16(ts.ts16).Sort(k, v);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx = tid * 16 + i;
        if (idx < n) {
            c[idx].vd = k[i];
            _unpack_diag(v[i], c[idx].k, c[idx].xo);
        }
    }
    __syncthreads();
}

// ── Public entry point ────────────────────────────────────────────────────────
// Replaces sort_gwf_diags(c, n) in gwfa.cu.
// IMPORTANT: fast_parallel_sort_cub and sort_gwf_diags_cub both declare
// __shared__ SortTempStorage_u64.  Because these two functions are never
// active simultaneously within the same kernel (sort_gwf_diags is called
// from gwf_diag_sort, fast_parallel_sort from gwf_dedup, in sequence), the
// compiler can alias the two __shared__ declarations to the same physical
// address if you move the declaration to the kernel level and pass it in.
// See the "advanced: single shared allocation" note at the bottom of this
// file for how to do that if shared memory pressure is critical.
__device__ __inline__
void sort_gwf_diags_cub(gwf_diag_t *c, int32_t n)
{
    __shared__ SortTempStorage_u64 _ts;         // union covers all IPT

    if      (n <=  32) _sort_diag_ipt1 (c, n, _ts);
    else if (n <=  64) _sort_diag_ipt2 (c, n, _ts);
    else if (n <= 128) _sort_diag_ipt4 (c, n, _ts);
    else if (n <= 256) _sort_diag_ipt8 (c, n, _ts);
    else               _sort_diag_ipt16(c, n, _ts);
}


// =============================================================================
//  HOW TO INTEGRATE INTO gwfa.cu
// =============================================================================
//
//  1. Add at the top of gwfa.cu (after the other includes):
//       #include "cub_sort_gwf.cuh"
//
//  2. In gwf_dedup(), replace:
//       if (!gwf_intv_is_sorted(cnt.n_tmp, buf->tmp, diag_shared))
//           fast_parallel_sort(buf->tmp.vd0, buf->tmp.vd1, cnt.n_tmp);
//     with:
//       if (!gwf_intv_is_sorted(cnt.n_tmp, buf->tmp, diag_shared))
//           fast_parallel_sort_cub(buf->tmp.vd0, buf->tmp.vd1, cnt.n_tmp);
//
//  3. In gwf_diag_sort(), replace the call:
//       sort_gwf_diags(c, n_c);
//     with:
//       sort_gwf_diags_cub(c, n_c);
//
//  4. The old fast_parallel_sort() and sort_gwf_diags() definitions can be
//     removed or left unused (the linker will drop them).
//
//
//  ADVANCED: SINGLE SHARED ALLOCATION (minimise shared memory pressure)
//  ─────────────────────────────────────────────────────────────────────
//  If occupancy profiling shows that having two separate __shared__
//  SortTempStorage_u64 declarations (one per function) adds too much
//  shared memory, hoist the single allocation to the kernel level:
//
//    In gwf_ed_kernel():
//        __shared__ SortTempStorage_u64 sort_ts;   // add this line
//        __shared__ uint64_t diag_shared[64];       // already present
//        ...
//        gwf_ed(g_d, ql, seq_ptr, 0, -1, 0, &local_buf, diag_shared, sort_ts);
//
//    Then thread sort_ts through gwf_ed → gwf_ed_extend → gwf_dedup →
//    fast_parallel_sort_cub / sort_gwf_diags_cub, passing it as a reference
//    parameter instead of declaring __shared__ inside each function.
//    This ensures both functions reuse the exact same physical shared memory.
//
//
//  REGISTER BUDGET NOTE
//  ─────────────────────
//  With IPT=16 each thread must hold 16 keys + 16 values = 32 uint64_t
//  registers = 256 bytes of register file.  Combined with the existing ~80
//  registers used by gwf_ed, this may push total register usage above the
//  threshold for 25 concurrent blocks per SM.  Profile with:
//      ncu --metrics launch__registers_per_thread ./gwf-test ...
//  and lower IPT_MAX to 8 (max 256 elements) if register spill appears.
//
// =============================================================================
