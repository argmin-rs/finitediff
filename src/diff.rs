// Copyright 2018 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::EPS_F64;

pub fn forward_diff_vec_f64(p: &Vec<f64>, f: &Fn(&Vec<f64>) -> f64) -> Vec<f64> {
    let fx = (f)(&p);
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            x1[i] += EPS_F64.sqrt();
            let fx1 = (f)(&x1);
            (fx1 - fx) / (EPS_F64.sqrt())
        })
        .collect()
}

#[cfg(feature = "ndarray")]
pub fn forward_diff_ndarray_f64(
    x: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
) -> ndarray::Array1<f64> {
    let fx = (f)(&x);
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut x1 = x.clone();
            x1[i] += EPS_F64.sqrt();
            let fx1 = (f)(&x1);
            (fx1 - fx) / (EPS_F64.sqrt())
        })
        .collect()
}

pub fn central_diff_vec_f64(x: &Vec<f64>, f: &Fn(&Vec<f64>) -> f64) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut x1 = x.clone();
            let mut x2 = x.clone();
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (f)(&x1);
            let fx2 = (f)(&x2);
            (fx1 - fx2) / (2.0 * EPS_F64.sqrt())
        })
        .collect()
}

#[cfg(feature = "ndarray")]
pub fn central_diff_ndarray_f64(
    x: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
) -> ndarray::Array1<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut x1 = x.clone();
            let mut x2 = x.clone();
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (f)(&x1);
            let fx2 = (f)(&x2);
            (fx1 - fx2) / (2.0 * EPS_F64.sqrt())
        })
        .collect()
}
