#include <stdint.h>
#include <stddef.h>
#include "gpu_common.h"

__device__ void dev_SHA512(const u8 *d, u64 n, u8 *md);

__global__ void global_SHA512(const u8 *d, u64 n, u8 *md);

__global__ void global_parallel_SHA512(const u8 *msg, u8 *md, u64 size, u64 n);
