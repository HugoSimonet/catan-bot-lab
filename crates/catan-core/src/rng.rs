//! Small, fast, `Copy` random number generator (SplitMix64).
//!
//! The game state embeds its RNG streams, so the generator must be `Copy` and
//! cheap to clone. SplitMix64 passes BigCrush and is more than adequate for
//! dice, shuffles and bot tie-breaking.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    state: u64,
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: mix(seed ^ GOLDEN),
        }
    }

    /// An independent stream derived from `seed`. Used to give the board,
    /// dice, dev deck, steals and each bot their own stream, so that a change
    /// in one consumer never shifts the random numbers seen by another.
    pub fn stream(seed: u64, stream: u64) -> Self {
        Rng::new(mix(seed.wrapping_add(mix(
            stream.wrapping_mul(GOLDEN) ^ 0xD1B5_4A32_D192_ED03
        ))))
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN);
        mix(self.state)
    }

    /// Uniform integer in `0..n` (n > 0). Uses the multiply-shift reduction;
    /// the bias is below 2^-32 and irrelevant here.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        (((self.next_u64() >> 32) * n as u64) >> 32) as u32
    }

    /// Uniform float in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            items.swap(i, j);
        }
    }

    /// Roll two six-sided dice, returning their sum.
    #[inline]
    pub fn roll_2d6(&mut self) -> u8 {
        (self.below(6) + self.below(6) + 2) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_deterministic_and_distinct() {
        let mut a = Rng::stream(42, 1);
        let mut b = Rng::stream(42, 1);
        let mut c = Rng::stream(42, 2);
        let xa: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let xb: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        let xc: Vec<u64> = (0..8).map(|_| c.next_u64()).collect();
        assert_eq!(xa, xb);
        assert_ne!(xa, xc);
    }

    #[test]
    fn dice_distribution_is_triangular() {
        let mut rng = Rng::new(7);
        let mut counts = [0u32; 13];
        let n = 360_000;
        for _ in 0..n {
            counts[rng.roll_2d6() as usize] += 1;
        }
        for (sum, ways) in [(2, 1), (7, 6), (12, 1), (6, 5), (8, 5)] {
            let expected = n as f64 * ways as f64 / 36.0;
            let got = counts[sum] as f64;
            assert!(
                (got - expected).abs() / expected < 0.05,
                "sum {sum}: {got} vs {expected}"
            );
        }
    }
}
