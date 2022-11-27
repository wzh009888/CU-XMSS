#include "gpu_wots.h"
#include "gpu_utils.h"
#include "gpu_hash.h"
#include "gpu_hash_address.h"
#include <iostream>
using namespace std;

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

__device__ void dev_expand_seed(u8 *outseeds, const u8 *inseed,
				const u8 *pub_seed, u32 addr[8])
{
	u32 i;
	u8 buf[N + 32];

	dev_set_hash_addr(addr, 0);
	dev_set_key_and_mask(addr, 0);
	memcpy(buf, pub_seed, N);
	for (i = 0; i < WOTS_LEN; i++) {
		dev_set_chain_addr(addr, i);
		dev_addr_to_bytes(buf + N, addr);
		dev_prf_keygen(outseeds + i * N, buf, inseed);
	}
} // dev_expand_seed

__device__ void dev_gen_chain(u8 *out, const u8 *in, u32 start, u32 steps,
			      const u8 *pub_seed, u32 addr[8])
{
	u32 i;

	/* Initialize out with the value at position 'start'. */
	memcpy(out, in, N);

	/* Iterate 'steps' calls to the hash function. */
	for (i = start; i < (start + steps) && i < WOTS_W; i++) {
		dev_set_hash_addr(addr, i);
		dev_thash_f(out, out, pub_seed, addr);
	}
} // dev_gen_chain

__device__ void dev_c_gen_chain(u8 *out, const u8 *in, u32 start, u32 steps,
				const u8 *pub_seed, u32 addr[8])
{
	/* Initialize out with the value at position 'start'. */
	unsigned char buf[PADDING_LEN + 2 * N];
	unsigned char bitmask[N];
	unsigned char addr_as_bytes[32];

	memcpy(out, in, N);
	dev_ull_to_bytes(buf, PADDING_LEN, 0);

	/* Iterate 'steps' calls to the hash function. */
	for (u32 i = start; i < (start + steps) && i < WOTS_W; i++) {
		dev_set_hash_addr(addr, i);

		dev_set_key_and_mask(addr, 0);
		dev_addr_to_bytes(addr_as_bytes, addr);
		dev_prf(buf + PADDING_LEN, addr_as_bytes, pub_seed);

		dev_set_key_and_mask(addr, 1);
		dev_addr_to_bytes(addr_as_bytes, addr);
		dev_prf(bitmask, addr_as_bytes, pub_seed);

		for (u32 j = 0; j < N; j++) {
			buf[PADDING_LEN + N + j] = out[j] ^ bitmask[j];
		}
		dev_core_hash(out, buf, PADDING_LEN + 2 * N);
	}
} // dev_gen_chain

__device__ void dev_base_w(int *output, const int out_len, const u8 *input)
{
	int in = 0;
	int out = 0;
	u8 total;
	int bits = 0;
	int consumed;

	for (consumed = 0; consumed < out_len; consumed++) {
		if (bits == 0) {
			total = input[in];
			in++;
			bits += 8;
		}
		bits -= WOTS_LOG_W;
		output[out] = (total >> bits) & (WOTS_W - 1);
		out++;
	}
} // dev_base_w

/* Computes the WOTS+ checksum over a message (in base_w). */
__device__ void dev_wots_checksum(int *csum_base_w, const int *msglobal_base_w)
{
	int csum = 0;
	u8 csum_bytes[(WOTS_LEN2 * WOTS_LOG_W + 7) / 8];
	u32 i;

	/* Compute checksum. */
	for (i = 0; i < WOTS_LEN1; i++)
		csum += WOTS_W - 1 - msglobal_base_w[i];

	/* Convert checksum to base_w. */
	/* Make sure expected empty zero bits are the least significant bits. */
	csum = csum << (8 - ((WOTS_LEN2 * WOTS_LOG_W) % 8));
	dev_ull_to_bytes(csum_bytes, sizeof(csum_bytes), csum);
	dev_base_w(csum_base_w, WOTS_LEN2, csum_bytes);

} // dev_wots_checksum

/* Takes a message and derives the matching chain lengths. */
__device__ void dev_chain_lengths(int *lengths, const u8 *msg)
{
	dev_base_w(lengths, WOTS_LEN1, msg);
	dev_wots_checksum(lengths + WOTS_LEN1, lengths);
} // dev_chain_lengths


__device__ void dev_wots_pkgen(u8 *pk, const u8 *seed,
			       const u8 *pub_seed, u32 addr[8])
{
	u32 i;

	/* The WOTS+ private key is derived from the seed. */
	dev_expand_seed(pk, seed, pub_seed, addr);

	for (i = 0; i < WOTS_LEN; i++) {
		dev_set_chain_addr(addr, i);
		dev_gen_chain(pk + i * N, pk + i * N, 0, WOTS_W - 1, pub_seed, addr);
	}
} // dev_wots_pkgen

__device__ void dev_wots_pkgen_parallel(u8 *pk, const u8 *seed,
					const u8 *pub_seed, u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();
	unsigned char buf[N + 32];

	g.sync();
	if (tid < WOTS_LEN) {
		dev_set_hash_addr(addr, 0);
		dev_set_key_and_mask(addr, 0);
		memcpy(buf, pub_seed, N);
		dev_set_chain_addr(addr, tid);
		dev_addr_to_bytes(buf + N, addr);
		// #ifdef USING_LOCAL_MEMORY
		// u8 temp[N];
		// dev_prf_keygen(temp, buf, seed);
		// dev_gen_chain(temp, temp, 0, WOTS_W - 1, pub_seed, addr);
		// memcpy(pk + tid * N, temp, N);
		// #else // ifdef USING_LOCAL_MEMORY
		dev_prf_keygen(pk + tid * N, buf, seed);
		dev_gen_chain(pk + tid * N, pk + tid * N,
			      0, WOTS_W - 1, pub_seed, addr);
		// #endif // ifdef USING_LOCAL_MEMORY
	}
} // dev_wots_pkgen_parallel

__device__ void dev_wots_sign(u8 *sig, const u8 *msg,
			      const u8 *seed, const u8 *pub_seed, u32 addr[8])
{
	int lengths[WOTS_LEN];
	u32 i;

	dev_chain_lengths(lengths, msg);

	/* The WOTS+ private key is derived from the seed. */
	dev_expand_seed(sig, seed, pub_seed, addr);

	for (i = 0; i < WOTS_LEN; i++) {
		dev_set_chain_addr(addr, i);
		dev_gen_chain(sig + i * N, sig + i * N, 0, lengths[i], pub_seed, addr);
	}
} // global_wots_sign

__device__ void dev_wots_sign_parallel(u8 *sig, const u8 *msg,  const u8 *seed,
				       const u8 *pub_seed, u32 addr[8], u32 offset)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	g.sync();
	if (tid >= offset && tid < offset + WOTS_LEN) {
		int lengths[WOTS_LEN];
		u8 buf[N + 32];
		int ttid = tid - offset;

		dev_chain_lengths(lengths, msg); // cannot be parallelized
		dev_set_hash_addr(addr, 0);
		dev_set_key_and_mask(addr, 0);
		memcpy(buf, pub_seed, N);
		dev_set_chain_addr(addr, ttid);
		dev_addr_to_bytes(buf + N, addr);
		// #ifdef USING_LOCAL_MEMORY
		// u8 temp[N];
		// dev_prf_keygen(temp, buf, seed);
		// dev_gen_chain(temp, temp, 0, lengths[ttid], pub_seed, addr);
		// memcpy(sig + ttid * N, temp, N);
		// #else // ifdef USING_LOCAL_MEMORY
		dev_prf_keygen(sig + ttid * N, buf, seed);
		dev_gen_chain(sig + ttid * N, sig + ttid * N,
			      0, lengths[ttid], pub_seed, addr);
		// #endif // ifdef USING_LOCAL_MEMORY
	}
} // dev_wots_sign_parallel

__device__ void dev_wots_pk_from_sig(u8 *pk, const u8 *sig,
				     const u8 *msg, const u8 *pub_seed, u32 addr[8])
{
	int lengths[WOTS_LEN];
	u32 i;

	dev_chain_lengths(lengths, msg);

	for (i = 0; i < WOTS_LEN; i++) {
		dev_set_chain_addr(addr, i);
		dev_gen_chain(pk + i * N, sig + i * N,
			      lengths[i], WOTS_W - 1 - lengths[i], pub_seed, addr);
	}
} // dev_wots_pk_from_sig

__device__ void dev_wots_pk_from_sig_parallel(u8 *pk, const u8 *sig,
					      const u8 *msg, const u8 *pub_seed, u32 addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	int lengths[WOTS_LEN];

	g.sync();
	if (tid < WOTS_LEN) {
		dev_chain_lengths(lengths, msg);
		dev_set_chain_addr(addr, tid);
		// #ifdef USING_LOCAL_MEMORY
		// u8 temp[N];
		// memcpy(temp, sig + tid * N, N);
		// dev_gen_chain(temp, temp, lengths[tid],
		// 	      WOTS_W - 1 - lengths[tid], pub_seed, addr);
		// memcpy(pk + tid * N, temp, N);
		// #else // ifdef USING_LOCAL_MEMORY
		dev_gen_chain(pk + tid * N, sig + tid * N, lengths[tid],
			      WOTS_W - 1 - lengths[tid], pub_seed, addr);
		// #endif // ifdef USING_LOCAL_MEMORY
	}
	g.sync();
} // dev_wots_pk_from_sig_parallel
