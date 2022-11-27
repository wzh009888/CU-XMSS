#include "gpu_common.h"

__device__ void dev_treehash(u8 *root, u8 *auth_path,
			     const u8 *sk_seed, const u8 *pub_seed,
			     u32 leaf_idx, const u32 subtree_addr[8]);

__device__ void dev_treehash_parallel_5(u8 *root, u8 *auth_path,
					const u8 *sk_seed, const u8 *pub_seed,
					u32 leaf_idx, const u32 subtree_addr[8]);
__device__ void dev_treehash_opk_parallel_5(u8 *root, u8 *auth_path,
					    const u8 *sk_seed, const u8 *pub_seed,
					    u64 idx_ex, u32 iter,
					    const u32 subtree_addr[8], int opk_num);
__device__ void dev_treehash_opk_multi_parallel_5(u8 *one_root, u8 *origin_sm,
						  const u8 *sk_seed, const u8 *pub_seed,
						  u64 origin_idx, u32 iter, u32 num,
						  u32 subtree_addr[8]);

__device__ void dev_treehash_parallel_10(u8 *root, u8 *auth_path,
					 const u8 *sk_seed, const u8 *pub_seed,
					 u32 leaf_idx, const u32 subtree_addr[8]);
__device__ void dev_treehash_opk_parallel_10(u8 *root, u8 *auth_path,
					     const u8 *sk_seed, const u8 *pub_seed,
					     u64 idx_ex, u32 iter,
					     const u32 subtree_addr[8], int opk_num);
__device__ void dev_treehash_opk_multi_parallel_10(u8 *one_root, u8 *origin_sm,
						   const u8 *sk_seed, const u8 *pub_seed,
						   u64 origin_idx, u32 iter, u32 num,
						   u32 subtree_addr[8]);

__device__ void dev_treehash_parallel_10_1(u8 *root, u8 *auth_path,
					   const u8 *sk_seed, const u8 *pub_seed,
					   u32 leaf_idx, const u32 subtree_addr[8]);
__device__ void dev_treehash_opk_parallel_10_1(u8 *root, u8 *auth_path,
					       const u8 *sk_seed, const u8 *pub_seed,
					       u64 idx_ex, u32 iter,
					       const u32 subtree_addr[8], int opk_num);

__device__ void dev_treehash_parallel_16(u8 *root, u8 *auth_path,
					 const u8 *sk_seed, const u8 *pub_seed,
					 u32 leaf_idx, const u32 subtree_addr[8]);
__device__ void dev_treehash_opk_parallel_16(u8 *root, u8 *auth_path,
					     const u8 *sk_seed, const u8 *pub_seed,
					     u64 idx_ex, u32 iter,
					     const u32 subtree_addr[8], int opk_num);

__device__ void dev_treehash_parallel_20(u8 *root, u8 *auth_path,
					 const u8 *sk_seed, const u8 *pub_seed,
					 u32 leaf_idx, const u32 subtree_addr[8]);
__device__ void dev_treehash_opk_parallel_20(u8 *root, u8 *auth_path,
					     const u8 *sk_seed, const u8 *pub_seed,
					     u64 idx_ex, u32 iter,
					     const u32 subtree_addr[8], int opk_num);
