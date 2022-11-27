#include "gpu_common.h"

__device__ void dev_addr_to_bytes(u8 *bytes, const u32 addr[8]);

__device__ int dev_core_hash(u8 *out, const u8 *in, u64 inlen);
__global__ void global_core_hash(u8 *out, const u8 *in, u64 inlen);

void face_sha256(const u8 *d, u64 n, u8 *md);
void face_dp_sha256(const u8 *msg, u8 *md, u64 size,
		    u64 msg_num, u64 blocks, u64 threads);
void face_msdp_sha256(const u8 *msg, u8 *md, u64 size,
		      u64 msg_num, u64 blocks, u64 threads);

void face_sha512(const u8 *d, u64 n, u8 *md);
void face_parallel_sha512(const u8 *msg, u8 *md, u64 size, u64 n);

void face_shake256(u8 *md, u64 outlen, const u8 *d, u64 inlen);
void face_parallel_shake256(const u8 *msg, u8 *md, u64 in_size, u64 out_size,
			    u64 msg_num, u64 blocks, u64 threads);

__device__ int dev_prf(u8 *out, const u8 in[32], const u8 *key);

__device__ int dev_prf_keygen(u8 *out, const u8 *in, const u8 *key);

__device__ int dev_hash_message(u8 *out, const u8 *R, const u8 *root,
				u64 idx, u8 *m_with_prefix, u64 mlen);
__device__ int dev_hash_message_modefied(u8 *out, const u8 *R, const u8 *root,
					 u64 idx, const u8 *m, u64 mlen);

__device__ int dev_thash_h(u8 *out, const u8 *in,
			   const u8 *pub_seed, u32 addr[8]);

__device__ int dev_thash_f(u8 *out, const u8 *in,
			   const u8 *pub_seed, u32 addr[8]);
// __device__ int dev_thash_f_parallel(u8 *out, const u8 *in,
// 				    const u8 *pub_seed, u32 addr[8], int t);
