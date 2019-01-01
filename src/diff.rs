// Copyright 2018 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::utils::*;
use crate::EPS_F64;

pub fn forward_diff_vec_f64(x: &Vec<f64>, f: &Fn(&Vec<f64>) -> f64) -> Vec<f64> {
    let fx = (f)(&x);
    let mut xt = x.clone();
    (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc_vec_f64(&mut xt, f, i, EPS_F64.sqrt());
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
    let mut xt = x.clone();
    (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc_ndarray_f64(&mut xt, f, i, EPS_F64.sqrt());
            (fx1 - fx) / (EPS_F64.sqrt())
        })
        .collect()
}

pub fn central_diff_vec_f64(x: &Vec<f64>, f: &Fn(&Vec<f64>) -> f64) -> Vec<f64> {
    let mut xt = x.clone();
    (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc_vec_f64(&mut xt, f, i, EPS_F64.sqrt());
            let fx2 = mod_and_calc_vec_f64(&mut xt, f, i, -EPS_F64.sqrt());
            (fx1 - fx2) / (2.0 * EPS_F64.sqrt())
        })
        .collect()
}

#[cfg(feature = "ndarray")]
pub fn central_diff_ndarray_f64(
    x: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
) -> ndarray::Array1<f64> {
    let mut xt = x.clone();
    (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc_ndarray_f64(&mut xt, f, i, EPS_F64.sqrt());
            let fx2 = mod_and_calc_ndarray_f64(&mut xt, f, i, -EPS_F64.sqrt());
            (fx1 - fx2) / (2.0 * EPS_F64.sqrt())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "ndarray")]
    use ndarray;

    const COMP_ACC: f64 = 1e-6;

    #[test]
    fn test_forward_diff_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2);
        let p = vec![1.0f64, 1.0f64];
        let grad = forward_diff_vec_f64(&p, &f);
        let res = vec![1.0f64, 2.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();

        let p = vec![1.0f64, 2.0f64];
        let grad = forward_diff_vec_f64(&p, &f);
        let res = vec![1.0f64, 4.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_forward_diff_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0f64]);

        let grad = forward_diff_ndarray_f64(&p, &f);
        let res = vec![1.0f64, 2.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();

        let p = ndarray::Array1::from_vec(vec![1.0f64, 2.0f64]);
        let grad = forward_diff_ndarray_f64(&p, &f);
        let res = vec![1.0f64, 4.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_central_diff_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2);
        let p = vec![1.0f64, 1.0f64];
        let grad = central_diff_vec_f64(&p, &f);
        let res = vec![1.0f64, 2.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();

        let p = vec![1.0f64, 2.0f64];
        let grad = central_diff_vec_f64(&p, &f);
        let res = vec![1.0f64, 4.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_central_diff_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0f64]);

        let grad = central_diff_ndarray_f64(&p, &f);
        let res = vec![1.0f64, 2.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();

        let p = ndarray::Array1::from_vec(vec![1.0f64, 2.0f64]);
        let grad = central_diff_ndarray_f64(&p, &f);
        let res = vec![1.0f64, 4.0];

        (0..2)
            .map(|i| assert!((res[i] - grad[i]).abs() < COMP_ACC))
            .count();
    }
}
