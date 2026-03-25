//! packed_simd-like wrapper layer

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[allow(non_camel_case_types)]
pub(crate) type i32x4 = __m128i;

pub(crate) fn new(e0: i32, e1: i32, e2: i32, e3: i32) -> i32x4 {
    unsafe { _mm_set_epi32(e3, e2, e1, e0) }
}

pub(crate) fn zero() -> i32x4 {
    unsafe { _mm_setzero_si128() }
}

pub(crate) fn extract(vals: i32x4, imm: usize) -> u32 {
    unsafe {
        match imm {
            0 => _mm_extract_epi32::<0>(vals) as u32,
            1 => _mm_extract_epi32::<1>(vals) as u32,
            2 => _mm_extract_epi32::<2>(vals) as u32,
            3 => _mm_extract_epi32::<3>(vals) as u32,
            _ => core::hint::unreachable_unchecked(),
        }
    }
}

pub(crate) fn insert(vals: &mut i32x4, val: i32, imm: usize) {
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
