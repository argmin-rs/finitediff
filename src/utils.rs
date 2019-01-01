// Copyright 2018 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[inline(always)]
pub fn mod_and_calc_vec_f64(x: &mut Vec<f64>, f: &Fn(&Vec<f64>) -> f64, idx: usize, y: f64) -> f64 {
    let xtmp = x[idx];
    x[idx] = xtmp + y;
    let fx1 = (f)(&x);
    x[idx] = xtmp;
    fx1
}

#[cfg(feature = "ndarray")]
#[inline(always)]
pub fn mod_and_calc_ndarray_f64(
    x: &mut ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
    idx: usize,
    y: f64,
) -> f64 {
    let xtmp = x[idx];
    x[idx] = xtmp + y;
    let fx1 = (f)(&x);
    x[idx] = xtmp;
    fx1
}
