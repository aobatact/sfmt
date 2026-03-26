//! packed_simd-like wrapper layer

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(non_camel_case_types)]
pub(crate) type i32x4 = __m128i;

#[cfg(target_arch = "wasm32")]
#[allow(non_camel_case_types)]
pub(crate) type i32x4 = v128;

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[allow(non_camel_case_types)]
pub(crate) type i32x4 = int32x4_t;

#[inline]
pub(crate) fn new(e0: i32, e1: i32, e2: i32, e3: i32) -> i32x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        unsafe { _mm_set_epi32(e3, e2, e1, e0) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let v = i32x4_replace_lane::<0>(i32x4_splat(0), e0);
        let v = i32x4_replace_lane::<1>(v, e1);
        let v = i32x4_replace_lane::<2>(v, e2);
        i32x4_replace_lane::<3>(v, e3)
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        let v = vdupq_n_s32(0);
        let v = vsetq_lane_s32::<0>(e0, v);
        let v = vsetq_lane_s32::<1>(e1, v);
        let v = vsetq_lane_s32::<2>(e2, v);
        vsetq_lane_s32::<3>(e3, v)
    }
}

#[inline]
pub(crate) fn zero() -> i32x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        unsafe { _mm_setzero_si128() }
    }

    #[cfg(target_arch = "wasm32")]
    {
        i32x4_splat(0)
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        vdupq_n_s32(0)
    }
}

#[inline]
pub(crate) fn extract(vals: i32x4, imm: usize) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        match imm {
            0 => _mm_extract_epi32::<0>(vals) as u32,
            1 => _mm_extract_epi32::<1>(vals) as u32,
            2 => _mm_extract_epi32::<2>(vals) as u32,
            3 => _mm_extract_epi32::<3>(vals) as u32,
            _ => core::hint::unreachable_unchecked(),
        }
    }

    #[cfg(target_arch = "wasm32")]
    match imm {
        0 => i32x4_extract_lane::<0>(vals) as u32,
        1 => i32x4_extract_lane::<1>(vals) as u32,
        2 => i32x4_extract_lane::<2>(vals) as u32,
        3 => i32x4_extract_lane::<3>(vals) as u32,
        _ => unsafe { core::hint::unreachable_unchecked() },
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        match imm {
            0 => vgetq_lane_s32::<0>(vals) as u32,
            1 => vgetq_lane_s32::<1>(vals) as u32,
            2 => vgetq_lane_s32::<2>(vals) as u32,
            3 => vgetq_lane_s32::<3>(vals) as u32,
            _ => core::hint::unreachable_unchecked(),
        }
    }
}

#[inline]
pub(crate) fn insert(vals: &mut i32x4, val: i32, imm: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let updated = unsafe {
            match imm {
                0 => _mm_insert_epi32::<0>(*vals, val),
                1 => _mm_insert_epi32::<1>(*vals, val),
                2 => _mm_insert_epi32::<2>(*vals, val),
                3 => _mm_insert_epi32::<3>(*vals, val),
                _ => core::hint::unreachable_unchecked(),
            }
        };
        unsafe {
            ::std::ptr::write(vals, updated);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        *vals = match imm {
            0 => i32x4_replace_lane::<0>(*vals, val),
            1 => i32x4_replace_lane::<1>(*vals, val),
            2 => i32x4_replace_lane::<2>(*vals, val),
            3 => i32x4_replace_lane::<3>(*vals, val),
            _ => unsafe { core::hint::unreachable_unchecked() },
        };
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    {
        *vals = unsafe {
            match imm {
                0 => vsetq_lane_s32::<0>(val, *vals),
                1 => vsetq_lane_s32::<1>(val, *vals),
                2 => vsetq_lane_s32::<2>(val, *vals),
                3 => vsetq_lane_s32::<3>(val, *vals),
                _ => core::hint::unreachable_unchecked(),
            }
        };
    }
}

/// XOR two 128-bit vectors.
#[inline]
pub(crate) fn xor(a: i32x4, b: i32x4) -> i32x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        unsafe { _mm_xor_si128(a, b) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        v128_xor(a, b)
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        veorq_s32(a, b)
    }
}

/// AND two 128-bit vectors.
#[inline]
pub(crate) fn and(a: i32x4, b: i32x4) -> i32x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        unsafe { _mm_and_si128(a, b) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        v128_and(a, b)
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        vandq_s32(a, b)
    }
}

/// Logical right shift each 32-bit lane by `IMM8` bits.
#[inline]
pub(crate) fn shr_epi32<const IMM8: i32>(v: i32x4) -> i32x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        unsafe { _mm_srli_epi32::<IMM8>(v) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        u32x4_shr(v, IMM8 as u32)
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        vreinterpretq_s32_u32(vshrq_n_u32::<IMM8>(vreinterpretq_u32_s32(v)))
    }
}

/// Left shift each 32-bit lane by `IMM8` bits.
#[inline]
pub(crate) fn shl_epi32<const IMM8: i32>(v: i32x4) -> i32x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        unsafe { _mm_slli_epi32::<IMM8>(v) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        i32x4_shl(v, IMM8 as u32)
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        vshlq_n_s32::<IMM8>(v)
    }
}

/// Byte-level right shift of the entire 128-bit register by `IMM8` bytes.
#[inline]
pub(crate) fn shr_si128<const IMM8: i32>(v: i32x4) -> i32x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        unsafe { _mm_srli_si128::<IMM8>(v) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let z = i32x4_splat(0);
        // IMM8 is 1 or 3 in all SFMT parameter sets.
        match IMM8 {
            1 => u8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(v, z),
            3 => u8x16_shuffle::<3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18>(v, z),
            _ => unreachable!(),
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        vreinterpretq_s32_u8(vextq_u8::<IMM8>(
            vreinterpretq_u8_s32(v),
            vdupq_n_u8(0),
        ))
    }
}

/// Byte-level left shift of the entire 128-bit register by `IMM8` bytes.
#[inline]
pub(crate) fn shl_si128<const IMM8: i32>(v: i32x4) -> i32x4 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        unsafe { _mm_slli_si128::<IMM8>(v) }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let z = i32x4_splat(0);
        // IMM8 is 1, 3, or 7 in all SFMT parameter sets.
        match IMM8 {
            1 => u8x16_shuffle::<0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30>(
                z, v,
            ),
            3 => u8x16_shuffle::<0, 1, 2, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28>(
                z, v,
            ),
            7 => u8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20, 21, 22, 23, 24>(z, v),
            _ => unreachable!(),
        }
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    unsafe {
        let v8 = vreinterpretq_u8_s32(v);
        let z8 = vdupq_n_u8(0);
        // IMM8 is 1, 3, or 7 in all SFMT parameter sets.
        vreinterpretq_s32_u8(match IMM8 {
            1 => vextq_u8::<15>(z8, v8),
            3 => vextq_u8::<13>(z8, v8),
            7 => vextq_u8::<9>(z8, v8),
            _ => unreachable!(),
        })
    }
}
