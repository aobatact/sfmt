//! packed_simd-like wrapper layer

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(non_camel_case_types)]
pub(crate) type i32x4 = __m128i;

#[cfg(target_arch = "wasm32")]
#[allow(non_camel_case_types)]
pub(crate) type i32x4 = v128;

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
}
