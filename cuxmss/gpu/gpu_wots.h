#include "gpu_common.h"

__device__ void dev_expand_seed(u8 *outseeds, const u8 *inseed,
				const u8 *pub_seed, u32 addr[8]);

__device__ void dev_gen_chain(u8 *out, const u8 *in, u32 start, u32 steps,
			      const u8 *pub_seed, u32 addr[8]);

__device__ void dev_c_gen_chain(u8 *out, const u8 *in, u32 start, u32 steps,
				const u8 *pub_seed, u32 addr[8]);

__device__ void dev_base_w(int *output, const int out_len, const u8 *input);

__device__ void dev_wots_checksum(int *csum_base_w, const int *msglobal_base_w);

__device__ void dev_chain_lengths(int *lengths, const u8 *msg);

__device__ void dev_wots_pkgen(u8 *pk, const u8 *seed,
			       const u8 *pub_seed, u32 addr[8]);

__device__ void dev_wots_pkgen_parallel(u8 *pk, const u8 *seed,
					const u8 *pub_seed, u32 addr[8]);

__device__ void dev_wots_sign(u8 *sig, const u8 *msg,
			      const u8 *seed, const u8 *pub_seed, u32 addr[8]);

__device__ void dev_wots_sign_parallel(u8 *sig, const u8 *msg,  const u8 *seed,
				       const u8 *pub_seed, u32 addr[8], u32 offset);

__device__ void dev_wots_pk_from_sig(u8 *pk, const u8 *sig,
				     const u8 *msg, const u8 *pub_seed, u32 addr[8]);

__device__ void dev_wots_pk_from_sig_parallel(u8 *pk, const u8 *sig,
					      const u8 *msg, const u8 *pub_seed, u32 addr[8]);
