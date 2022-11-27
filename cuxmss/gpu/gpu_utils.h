#include "gpu_common.h"

__device__ void dev_ull_to_bytes(u8 *out, u32 outlen, u64 in);

__global__ void global_ull_to_bytes(u8 *out, u32 outlen, u64 *in);

void face_ull_to_bytes(u8 *out, u32 outlen, u64 in);

__device__ u64 dev_bytes_to_ull(const u8 *in, u32 inlen);
