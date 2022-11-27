#include "gpu_common.h"

int gpu_xmss_keypair_fast(u8 *pk, u8 *sk);
int gpu_xmssmt_keypair_fast(u8 *pk, u8 *sk);

int gpu_xmss_dp_keypair_fast(u8 *pk, u8 *sk, u32 num);
int gpu_xmssmt_dp_keypair_fast(u8 *pk, u8 *sk, u32 num);

int gpu_xmss_ip_keypair_fast(u8 *pk, u8 *sk);
int gpu_xmssmt_ip_keypair_fast(u8 *pk, u8 *sk);
