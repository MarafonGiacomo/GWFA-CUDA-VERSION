# GWFA CUDA Version

CUDA-accelerated version of GWFA (Graph WaveFront Alignment), adapted from the original proof-of-concept implementation by Heng Li.

GWFA aligns a sequence against a sequence graph and computes the edit distance without backtracing. This repository focuses on accelerating the core alignment procedure on NVIDIA GPUs using CUDA.

This is an experimental academic implementation, not an end-user read mapping tool.

## Overview

The project ports and optimizes the GWFA algorithm for GPU execution.

The implementation uses two levels of parallelism:

- different reads are processed independently by different CUDA blocks;
- the 32 threads inside each block cooperate on the alignment of a single read.

Main features:

- CUDA implementation of GWFA edit-distance computation;
- one-warp-per-read execution model;
- warp-level acceleration of the extension phase;
- optimized state management, sorting and deduplication;
- CUDA streams for batched execution;
- support for GFA graph inputs and FASTA query sequences.

## Getting Started

Clone the repository and build the project:

```sh
git clone https://github.com/MarafonGiacomo/GWFA-CUDA-VERSION
cd GWFA-CUDA-VERSION
make
```

Run a test with a generated graph and query file:

```sh
./gwf-test exposed_testing/generated/grafo_test.gfa exposed_testing/generated/ref1k_1.fa
```

Other query files can be tested by replacing the `.fa` input:

```sh
./gwf-test exposed_testing/generated/grafo_test.gfa exposed_testing/generated/ref1k_5.fa
./gwf-test exposed_testing/generated/grafo_test.gfa exposed_testing/generated/ref1k_10.fa
./gwf-test exposed_testing/generated/grafo_test.gfa exposed_testing/generated/ref5k_1.fa
./gwf-test exposed_testing/generated/grafo_test.gfa exposed_testing/generated/ref10k_1.fa
```

## Test Files

The generated test files follow this naming convention:

```text
ref<read_length>_<error_rate>.fa
```

Examples:

```text
ref1k_1.fa   -> reads of length 1k with 1% error rate
ref1k_5.fa   -> reads of length 1k with 5% error rate
ref5k_10.fa  -> reads of length 5k with 10% error rate
ref10k_1.fa  -> reads of length 10k with 1% error rate
```

## Implementation Notes

Each read is assigned to one CUDA block. Since each block uses 32 threads, the internal execution model maps naturally to one warp per read.

The extension phase is accelerated with warp-level primitives. Each CUDA lane compares one graph/query character pair, and `__ballot_sync()` is used to detect mismatches across the warp. When all characters match, the algorithm can advance by 32 positions in one iteration.

CUDA streams are used to process batches of reads and reduce idle time between memory transfers and kernel execution.

## Benchmark Summary

The implementation was tested on a laptop-class CUDA platform:

```text
CPU: AMD Ryzen 5 7645HX
GPU: NVIDIA GeForce RTX 4050, 6 GB
```

The CUDA version achieves the highest speedups on shorter reads and lower error rates. Performance decreases as read length and error rate increase, because the wavefronts become larger and graph traversal, sorting, deduplication and memory traffic become more expensive.

## Limitations

This implementation follows the same general assumptions as the original GWFA proof of concept:

- it computes edit distance only;
- it does not perform backtracing;
- it is not a complete read mapper;
- it should be tested carefully on complex graph corner cases.

## Original GWFA

The original GWFA repository is available here:

https://github.com/lh3/gwfa

The development of GWFA has since moved to `gfatools`, where the algorithm can operate directly on bidirected sequence graphs.

## Attribution

This project is based on the original GWFA implementation and adapts it for CUDA-based acceleration.
