#include "gpu_hash_address.h"

__device__ void dev_set_layer_addr(u32 addr[8], u32 layer)
{
	addr[0] = layer;
} // dev_set_layer_addr

__device__ void dev_set_tree_addr(u32 addr[8], u64 tree)
{
	addr[1] = (u32)(tree >> 32);
	addr[2] = (u32)tree;
} // dev_set_tree_addr

__device__ void dev_set_type(u32 addr[8], u32 type)
{
	addr[3] = type;
} // dev_set_type

__device__ void dev_set_key_and_mask(u32 addr[8], u32 key_and_mask)
{
	addr[7] = key_and_mask;
} // dev_set_key_and_mask

__device__ void dev_copy_subtree_addr(u32 out[8], const u32 in[8])
{
	out[0] = in[0];
	out[1] = in[1];
	out[2] = in[2];
} // dev_copy_subtree_addr

/* These functions are used for OTS addresses. */

__device__ void dev_set_ots_addr(u32 addr[8], u32 ots)
{
	addr[4] = ots;
} // dev_set_ots_addr

__device__ void dev_set_chain_addr(u32 addr[8], u32 chain)
{
	addr[5] = chain;
} // dev_set_chain_addr

__device__ void dev_set_hash_addr(u32 addr[8], u32 hash)
{
	addr[6] = hash;
} // dev_set_hash_addr

/* This function is used for L-tree addresses. */

__device__ void dev_set_ltree_addr(u32 addr[8], u32 ltree)
{
	addr[4] = ltree;
} // dev_set_ltree_addr

/* These functions are used for hash tree addresses. */

__device__ void dev_set_tree_height(u32 addr[8], u32 tree_height)
{
	addr[5] = tree_height;
} // dev_set_tree_height

__device__ void dev_set_tree_index(u32 addr[8], u32 tree_index)
{
	addr[6] = tree_index;
} // dev_set_tree_index
