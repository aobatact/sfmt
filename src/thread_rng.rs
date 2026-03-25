//! Thread-local RNG based on SFMT

use super::SFMT;

use core::convert::Infallible;
use rand_core::{Rng, SeedableRng, TryRng};
use std::cell::RefCell;
use std::rc::Rc;

thread_local!(
    static THREAD_RNG_KEY: Rc<RefCell<SFMT>> = {
        Rc::new(RefCell::new(
            SFMT::try_from_rng(&mut getrandom::SysRng).unwrap()
        ))
    }
);

/// Thread-local RNG based on SFMT.
///
/// See the reference of the function [thread_rng](fn.thread_rng.html), which generates this struct.
#[derive(Clone)]
pub struct ThreadRng {
    rng: Rc<RefCell<SFMT>>,
}

/// Create a thread local RNG.
///
/// The seed of SFMT is generated from OS randomness on each thread.
///
/// ```
/// # extern crate sfmt;
/// # extern crate rand;
/// # use rand::RngExt;
/// let mut rng = sfmt::thread_rng();
/// rng.random::<u32>(); // random u32
/// ```
pub fn thread_rng() -> ThreadRng {
    ThreadRng {
        rng: THREAD_RNG_KEY.with(|t| t.clone()),
    }
}

impl TryRng for ThreadRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.rng.borrow_mut().next_u32())
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.rng.borrow_mut().next_u64())
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Infallible> {
        self.rng.borrow_mut().fill_bytes(dest);
        Ok(())
    }
}
