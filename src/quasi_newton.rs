use ndarray::{arr1, Array1, Array2, ArrayView1};
use crate::gauss_seidel::gauss_seidel;
use crate::newton::FuncX;

/// Secant method
///
/// returns the root of f(x).
///
/// * `x_0` - initial value for recurrence relation.
/// * `x_1` - initial value for recurrence relation.
/// * `f` - f(x).
/// * `eps` - error threshold.
pub fn secant(f: &fn(f64) -> f64, x_0: f64, x_1: f64, eps: f64) -> f64 {
    // define x0, x1, x2
    let mut x0 = x_0;
    let mut x1 = x_1;

    //update x0, x1
    loop {
        let x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0));
        println!("{}", x2);
        x0 = x1;
        x1 = x2;
        // if convergence, return x2
        if f64::abs(x0 - x1) < eps {
            return x2;
        }
    }
}

/// Return Newton step (alpha)
///
/// * `f` - f(x).
/// * `del_f` - del(f)
/// * `p` - gradient decent vector.
/// * `x_0` - start point.
pub fn line_search(f: FuncX, del_f: Vec<&FuncX>, p: ArrayView1<f64>, x_0: ArrayView1<f64>) -> f64 {
    let n = x_0.len();
    let mut del_f0 = Array1::zeros(n);
    for i in 0..n {
        del_f0[i] = del_f[i](x_0);
    }

    let c1 = 1e-4;
    let m = del_f0.dot(&p);
    let t = - c1 * m;
    let rho = 0.5;
    let mut alpha = 10.0;

    loop {
        let x_1 = &x_0 + (Array1::from_elem(n, alpha) * &p);
        if f(x_0) - f(x_1.view()) >= alpha * t {
            return alpha
        }
        alpha *= rho;
        println!("{}", alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use core::f64;
    use ndarray::ArrayView1;
    use crate::assert::norm_l2;

    #[test]
    fn line_search_1d() {
        // prepare
        let x_0 = arr1(&[0.0]);
        let p = arr1(&[2.0]);
        let n = x_0.len();
        let f = |x: ArrayView1<f64>| x[0].powf(2.0) - 2.0 * x[0] + 1.0;
        let f_x: FuncX = |x: ArrayView1<f64>| 2.0 * x[0] - 2.0;
        let del_f = vec![&f_x];

        // calculate alpha (inexact line search)
        let alpha = line_search(f, del_f, p.view(), x_0.view());

        // assert f(x + alpha * p) < f(x)
        let x_1 = &x_0 + (Array1::from_elem(n, alpha) * &p);
        println!("alpha={}, f(x0)={}, x1={}, f(x1)={}", alpha, f(x_0.view()), x_1, f(x_1.view()));
        assert!(f(x_1.view()) < f(x_0.view()));
    }

    #[test]
    fn secant_sin_cos() {
        let eps = 1e-10;
        // arrange f(x), x0, x1, n
        let f: fn(f64) -> f64 = |x: f64| f64::sin(x / 4.0) - f64::cos(x / 4.0);
        let x_0 = 0.0;
        let x_1 = 8.0;
        // calculate x using secant-method
        let x = secant(&f, x_0, x_1, eps);
        // x approximately equals to PI.
        assert!(std::f64::consts::PI - x < 1e-10);
    }
}
