// Copyright 2018 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::utils::*;
use crate::EPS_F64;

pub fn forward_hessian_vec_f64(x: &Vec<f64>, grad: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let fx = (grad)(x);
    let mut xt = x.clone();
    let out: Vec<Vec<f64>> = (0..x.len())
        .map(|i| {
            let fx1 = mod_and_calc_vec_f64(&mut xt, grad, i, EPS_F64.sqrt());
            fx1.iter()
                .zip(fx.iter())
                .map(|(a, b)| (a - b) / (EPS_F64.sqrt()))
                .collect::<Vec<f64>>()
        })
        .collect();

    // restore symmetry
    restore_symmetry_vec_f64(out)
}

#[cfg(feature = "ndarray")]
pub fn forward_hessian_ndarray_f64(
    x: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    // use ndarray::s;
    let mut xt = x.clone();
    let fx = (grad)(&x);
    let rn = fx.len();
    let n = x.len();
    let mut out = unsafe { ndarray::Array2::uninitialized((n, rn)) };
    for i in 0..n {
        let fx1 = mod_and_calc_ndarray_f64(&mut xt, grad, i, EPS_F64.sqrt());
        // unfortunately, this is slower than iterating :/
        // out.slice_mut(s![i, ..])
        //     .assign(&((fx1 - &fx) / EPS_F64.sqrt()));
        for j in 0..rn {
            out[(i, j)] = (fx1[j] - fx[j]) / EPS_F64.sqrt();
        }
    }
    // restore symmetry
    restore_symmetry_ndarray_f64(out)
}

pub fn central_hessian_vec_f64(x: &Vec<f64>, grad: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let n = x.len();
    let out: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut x1 = x.clone();
            let mut x2 = x.clone();
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (grad)(&x1);
            let fx2 = (grad)(&x2);
            fx1.iter()
                .zip(fx2.iter())
                .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
                .collect::<Vec<f64>>()
        })
        .collect();

    // restore symmetry
    restore_symmetry_vec_f64(out)
}

#[cfg(feature = "ndarray")]
pub fn central_hessian_ndarray_f64(
    x: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    // TODO: get rid of this!
    let fx = (grad)(&x);
    let rn = fx.len();
    let n = x.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = x.clone();
        let mut x2 = x.clone();
        x1[i] += EPS_F64.sqrt();
        x2[i] -= EPS_F64.sqrt();
        let fx1 = (grad)(&x1);
        let fx2 = (grad)(&x2);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx2[j]) / (2.0 * EPS_F64.sqrt());
        }
    }
    // restore symmetry
    restore_symmetry_ndarray_f64(out)
}

pub fn forward_hessian_vec_prod_vec_f64(
    x: &Vec<f64>,
    grad: &Fn(&Vec<f64>) -> Vec<f64>,
    p: &Vec<f64>,
) -> Vec<f64> {
    let fx = (grad)(x);
    let out: Vec<f64> = {
        let x1 = x
            .iter()
            .zip(p.iter())
            .map(|(xi, pi)| xi + pi * EPS_F64.sqrt())
            .collect();
        let fx1 = (grad)(&x1);
        fx1.iter()
            .zip(fx.iter())
            .map(|(a, b)| (a - b) / (EPS_F64.sqrt()))
            .collect::<Vec<f64>>()
    };
    out
}

#[cfg(feature = "ndarray")]
pub fn forward_hessian_vec_prod_ndarray_f64(
    x: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    p: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let fx = (grad)(&x);
    let rn = fx.len();
    let mut out = ndarray::Array1::zeros(rn);
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi + pi * EPS_F64.sqrt())
        .collect();
    let fx1 = (grad)(&x1);
    for j in 0..rn {
        out[j] = (fx1[j] - fx[j]) / EPS_F64.sqrt();
    }
    out
}

pub fn central_hessian_vec_prod_vec_f64(
    x: &Vec<f64>,
    grad: &Fn(&Vec<f64>) -> Vec<f64>,
    p: &Vec<f64>,
) -> Vec<f64> {
    let out: Vec<f64> = {
        let x1 = x
            .iter()
            .zip(p.iter())
            .map(|(xi, pi)| xi + pi * EPS_F64.sqrt())
            .collect();
        let x2 = x
            .iter()
            .zip(p.iter())
            .map(|(xi, pi)| xi - pi * EPS_F64.sqrt())
            .collect();
        let fx1 = (grad)(&x1);
        let fx2 = (grad)(&x2);
        fx1.iter()
            .zip(fx2.iter())
            .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
            .collect::<Vec<f64>>()
    };
    out
}

#[cfg(feature = "ndarray")]
pub fn central_hessian_vec_prod_ndarray_f64(
    x: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    p: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let rn = x.len();
    let mut out = ndarray::Array1::zeros(rn);
    let x1 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi + pi * EPS_F64.sqrt())
        .collect();
    let x2 = x
        .iter()
        .zip(p.iter())
        .map(|(xi, pi)| xi - pi * EPS_F64.sqrt())
        .collect();
    let fx1 = (grad)(&x1);
    let fx2 = (grad)(&x2);
    for j in 0..rn {
        out[j] = (fx1[j] - fx2[j]) / (2.0 * EPS_F64.sqrt());
    }
    out
}

pub fn forward_hessian_nograd_vec_f64(x: &Vec<f64>, f: &Fn(&Vec<f64>) -> f64) -> Vec<Vec<f64>> {
    // TODO: f(x + EPS * e_i) needs to be precomputed!!
    let fx = (f)(x);
    let n = x.len();
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let t = {
                let mut xi = x.clone();
                xi[i] += EPS_F64.sqrt();
                let mut xj = x.clone();
                xj[j] += EPS_F64.sqrt();
                let mut xij = x.clone();
                xij[i] += EPS_F64.sqrt();
                xij[j] += EPS_F64.sqrt();
                let fxi = (f)(&xi);
                let fxj = (f)(&xj);
                let fxij = (f)(&xij);
                (fxij - fxi - fxj + fx) / EPS_F64
            };
            out[i][j] = t;
            out[j][i] = t;
        }
    }
    out
}

#[cfg(feature = "ndarray")]
pub fn forward_hessian_nograd_ndarray_f64(
    x: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
) -> ndarray::Array2<f64> {
    // TODO: f(x + EPS * e_i) needs to be precomputed!!
    let fx = (f)(x);
    let n = x.len();
    let mut out = ndarray::Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let t = {
                let mut xi = x.clone();
                xi[i] += EPS_F64.sqrt();
                let mut xj = x.clone();
                xj[j] += EPS_F64.sqrt();
                let mut xij = x.clone();
                xij[i] += EPS_F64.sqrt();
                xij[j] += EPS_F64.sqrt();
                let fxi = (f)(&xi);
                let fxj = (f)(&xj);
                let fxij = (f)(&xij);
                (fxij - fxi - fxj + fx) / EPS_F64
            };
            out[(i, j)] = t;
            out[(j, i)] = t;
        }
    }
    out
}

pub fn forward_hessian_nograd_sparse_vec_f64(
    x: &Vec<f64>,
    f: &Fn(&Vec<f64>) -> f64,
    indices: Vec<(usize, usize)>,
) -> Vec<Vec<f64>> {
    let fx = (f)(x);
    let n = x.len();
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for (i, j) in indices {
        let t = {
            let mut xi = x.clone();
            xi[i] += EPS_F64.sqrt();
            let mut xj = x.clone();
            xj[j] += EPS_F64.sqrt();
            let mut xij = x.clone();
            xij[i] += EPS_F64.sqrt();
            xij[j] += EPS_F64.sqrt();
            let fxi = (f)(&xi);
            let fxj = (f)(&xj);
            let fxij = (f)(&xij);
            (fxij - fxi - fxj + fx) / EPS_F64
        };
        out[i][j] = t;
        out[j][i] = t;
    }
    out
}

#[cfg(feature = "ndarray")]
pub fn forward_hessian_nograd_sparse_ndarray_f64(
    x: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
    indices: Vec<(usize, usize)>,
) -> ndarray::Array2<f64> {
    let fx = (f)(x);
    let n = x.len();
    let mut out = ndarray::Array2::zeros((n, n));
    for (i, j) in indices {
        let t = {
            let mut xi = x.clone();
            xi[i] += EPS_F64.sqrt();
            let mut xj = x.clone();
            xj[j] += EPS_F64.sqrt();
            let mut xij = x.clone();
            xij[i] += EPS_F64.sqrt();
            xij[j] += EPS_F64.sqrt();
            let fxi = (f)(&xi);
            let fxj = (f)(&xj);
            let fxij = (f)(&xij);
            (fxij - fxi - fxj + fx) / EPS_F64
        };
        out[(i, j)] = t;
        out[(j, i)] = t;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::*;
    #[cfg(feature = "ndarray")]
    use ndarray;

    const COMP_ACC: f64 = 1e-6;

    #[test]
    fn test_forward_hessian_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = forward_hessian_vec_f64(&p, &|d| forward_diff_vec_f64(&d, &f));
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_forward_hessian_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = forward_hessian_ndarray_f64(&p, &|d| forward_diff_ndarray_f64(&d, &f));
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_central_hessian_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = central_hessian_vec_f64(&p, &|d| central_diff_vec_f64(&d, &f));
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_central_hessian_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = central_hessian_ndarray_f64(&p, &|d| central_diff_ndarray_f64(&d, &f));
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }
    #[test]
    fn test_forward_hessian_vec_prod_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let x = vec![1.0f64, 1.0, 1.0, 1.0];
        let p = vec![2.0, 3.0, 4.0, 5.0];
        let hessian = forward_hessian_vec_prod_vec_f64(&x, &|d| forward_diff_vec_f64(&d, &f), &p);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_forward_hessian_vec_prod_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let x = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let p = ndarray::Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let hessian =
            forward_hessian_vec_prod_ndarray_f64(&x, &|d| forward_diff_ndarray_f64(&d, &f), &p);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_central_hessian_vec_prod_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let x = vec![1.0f64, 1.0, 1.0, 1.0];
        let p = vec![2.0, 3.0, 4.0, 5.0];
        let hessian = central_hessian_vec_prod_vec_f64(&x, &|d| forward_diff_vec_f64(&d, &f), &p);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_central_hessian_vec_prod_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let x = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let p = ndarray::Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let hessian =
            central_hessian_vec_prod_ndarray_f64(&x, &|d| forward_diff_ndarray_f64(&d, &f), &p);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            assert!((res[i] - hessian[i]).abs() < COMP_ACC)
        }
    }

    #[test]
    fn test_forward_hessian_nograd_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = forward_hessian_nograd_vec_f64(&p, &f);
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_forward_hessian_nograd_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = forward_hessian_nograd_ndarray_f64(&p, &f);
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }

    #[test]
    fn test_forward_hessian_nograd_sparse_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let indices = vec![(1, 1), (2, 3), (3, 3)];
        let hessian = forward_hessian_nograd_sparse_vec_f64(&p, &f, indices);
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC)
            }
        }
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_forward_hessian_nograd_sparse_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let indices = vec![(1, 1), (2, 3), (3, 3)];
        let hessian = forward_hessian_nograd_sparse_ndarray_f64(&p, &f, indices);
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        for i in 0..4 {
            for j in 0..4 {
                assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC)
            }
        }
    }
}
