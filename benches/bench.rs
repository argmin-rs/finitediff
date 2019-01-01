// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Benches

#![feature(test)]

extern crate finitediff;
extern crate test;

const MASSIVENESS: usize = 512;

fn cost_vec_f64(x: &Vec<f64>) -> f64 {
    x.iter().fold(0.0, |a, acc| a + acc)
}

#[cfg(feature = "ndarray")]
fn cost_ndarray_f64(x: &ndarray::Array1<f64>) -> f64 {
    x.iter().fold(0.0, |a, acc| a + acc)
}

fn cost_multi_vec_f64(x: &Vec<f64>) -> Vec<f64> {
    x.clone()
}

#[cfg(feature = "ndarray")]
fn cost_multi_ndarray_f64(x: &ndarray::Array1<f64>) -> ndarray::Array1<f64> {
    x.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use finitediff::*;
    use test::{black_box, Bencher};

    #[bench]
    fn cost_func_vec_f64_1(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(cost_vec_f64(&x));
        });
    }

    #[bench]
    fn cost_func_vec_f64_np1(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            for _ in 0..(MASSIVENESS + 1) {
                black_box(cost_vec_f64(&x));
            }
        });
    }

    #[bench]
    fn cost_func_vec_f64_2n(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            for _ in 0..(2 * MASSIVENESS) {
                black_box(cost_vec_f64(&x));
            }
        });
    }

    #[bench]
    fn cost_func_multi_vec_f64_1(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(cost_multi_vec_f64(&x));
        });
    }

    #[bench]
    fn cost_func_multi_vec_f64_np1(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            for _ in 0..(MASSIVENESS + 1) {
                black_box(cost_multi_vec_f64(&x));
            }
        });
    }

    #[bench]
    fn cost_func_multi_vec_f64_2n(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            for _ in 0..(2 * MASSIVENESS) {
                black_box(cost_multi_vec_f64(&x));
            }
        });
    }

    #[bench]
    fn forward_diff_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.forward_diff(&cost_vec_f64));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn forward_diff_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from_vec(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.forward_diff(&cost_ndarray_f64));
        });
    }

    #[bench]
    fn central_diff_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.central_diff(&cost_vec_f64));
        });
    }

    #[cfg(feature = "ndarray")]
    #[bench]
    fn central_diff_ndarray_f64(b: &mut Bencher) {
        let x = ndarray::Array1::from_vec(vec![1.0f64; MASSIVENESS]);
        b.iter(|| {
            black_box(x.central_diff(&cost_ndarray_f64));
        });
    }

    #[bench]
    fn forward_jacobian_vec_f64(b: &mut Bencher) {
        let x = vec![1.0f64; MASSIVENESS];
        b.iter(|| {
            black_box(x.forward_jacobian(&cost_multi_vec_f64));
        });
    }
}
