#include "gpu_common.h"

int gpu_xmss_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
		       const u8 *m, u64 mlen);
int gpu_xmssmt_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			 const u8 *m, u64 mlen);

int gpu_xmss_dp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			  const u8 *m, u64 mlen, u64 num);
int gpu_xmssmt_dp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			    const u8 *m, u64 mlen, u64 num);

int gpu_xmss_msdp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			    const u8 *m, u64 mlen, u64 num);
int gpu_xmssmt_msdp_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			      const u8 *m, u64 mlen, u64 num);

int gpu_xmss_ip_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			  const u8 *m, u64 mlen);
int gpu_xmssmt_ip_sign_fast(u8 *sk, u8 *sm, u64 *smlen,
			    const u8 *m, u64 mlen);
