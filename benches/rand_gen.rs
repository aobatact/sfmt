#![feature(test)]

extern crate rand;
extern crate sfmt;
extern crate test;

use rand::*;
use rand_core::SeedableRng;
use rand_xorshift::*;
use sfmt::SFMT;
use test::Bencher;

macro_rules! def_bench {
    ($name:ident, $t:ty, $rng:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut rng = $rng;
            b.iter(|| {
                for _ in 0..100 {
                    let _rng = rng.random::<$t>();
                }
            });
        }
    };
} // def_bench!

mod gen_f64 {
    use super::*;
    def_bench!(xorshift, f64, XorShiftRng::seed_from_u64(0));
    def_bench!(sfmt, f64, SFMT::seed_from_u64(0));
}

mod gen_f32 {
    use super::*;
    def_bench!(xorshift, f32, XorShiftRng::seed_from_u64(0));
    def_bench!(sfmt, f32, SFMT::seed_from_u64(0));
}

mod gen_u64 {
    use super::*;
    def_bench!(xorshift, u64, XorShiftRng::seed_from_u64(0));
    def_bench!(sfmt, u64, SFMT::seed_from_u64(0));
}

mod gen_u32 {
    use super::*;
    def_bench!(xorshift, u32, XorShiftRng::seed_from_u64(0));
    def_bench!(sfmt, u32, SFMT::seed_from_u64(0));
}
