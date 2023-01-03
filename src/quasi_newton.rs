use crate::assert::norm_l2;

use crate::newton::FuncX;
use ndarray::{Array1, Array2, ArrayView1};


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

/// Quasi-Newton method
///
/// returns local minimum of f(x).
///
/// * `x_0` - start point.
/// * `del_f` - del(f).
/// * `b` - Approximate Hessian matrix of f(x).
/// * `f` - f(x).
/// * `eps` - error threshold.
pub fn bfgs(f: FuncX, del_f: Vec<&FuncX>, x_0: ArrayView1<f64>) -> Array1<f64> {
    let delta = 1e-8;
    let n = x_0.len();
    let mut h: Array2<f64> = Array2::eye(n);
    let mut x_k1 = x_0.to_owned();
    let mut del_f_k_ = del(&del_f, x_k1.view());

    loop {
        // calculate del(f(x_k))

        // Calculate gradient descent p, with p = - H * del(f)
        let p = -&h.dot(&del_f_k_);

        let alpha = line_search(f, &del_f, p.view(), x_k1.view());
        let s = alpha * &p;
        x_k1 += &s;
        // Calculate del(f(x_k+1)) - del(f(x_k))
        // Because del(f) < 0, y > 0 should be satisfied.
        let del_f_k1 = del(&del_f, x_k1.view());
        let y = &del_f_k1 - &del_f_k_;

        let rho = 1.0 / &y.dot(&s);
        let lhm = Array2::eye(n) - rho * x_yt(s.view(), y.view());
        let rhm = Array2::eye(n) - rho * x_yt(y.view(), s.view());
        h = lhm.dot(&h.view()).dot(&rhm.view()) + rho * x_yt(s.view(), s.view());
        del_f_k_ = del_f_k1;

        if f64::abs(norm_l2(del(&del_f, x_k1.view()).view())) < delta {
            return x_k1;
        }
    }
}

/// Return Newton step (alpha)
///
/// * `f` - f(x).
/// * `del_f` - del(f)
/// * `p` - gradient descent vector.
/// * `x_0` - start point.
pub fn line_search(f: FuncX, del_f: &Vec<&FuncX>, p: ArrayView1<f64>, x_0: ArrayView1<f64>) -> f64 {
    let rho = 0.5;
    let c1 = 1e-4;
    let c2 = 0.9;
    let mut alpha = 10.0;

    loop {
        let x_1 = axpy(alpha, p, x_0);
        println!("x_0={}, x_1={}", x_0, x_1);

        if is_satisfied_wolfe_conditions(f, &del_f, p, alpha, c1, c2, x_0, x_1.view()) {
            return alpha;
        }
        alpha *= rho;
        println!("{}", alpha);
    }
}

/// Calculate ax + y
///
/// * `a` - scalar.
/// * `x` - vector.
/// * `y` - vector.
pub fn axpy(a: f64, x: ArrayView1<f64>, y: ArrayView1<f64>) -> Array1<f64> {
    let n = x.len();
    let mut vec = Array1::zeros(n);
    for i in 0..n {
        vec[i] = a * x[i] + y[i];
    }

    vec
}

/// Calculate ax
///
/// * `a` - scalar.
/// * `x` - vector.
pub fn ax(a: f64, x: ArrayView1<f64>) -> Array1<f64> {
    let n = x.len();
    let mut vec = Array1::zeros(n);
    for i in 0..n {
        vec[i] = a * x[i];
    }

    vec
}

/// Calculate x y^T
///
/// Return N x N matrix.
/// * `x` - N-elements vector.
/// * `y` - N-elements vector.
pub fn x_yt(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Array2<f64> {
    let m = x.len();
    let n = y.len();
    let mut matrix = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            matrix[[i, j]] = x[i] * y[j];
        }
    }

    matrix
}

pub fn del(del_f: &Vec<&FuncX>, x: ArrayView1<f64>) -> Array1<f64> {
    let n = del_f.len();
    let mut del = Array1::zeros(n);
    for i in 0..n {
        del[i] = del_f[i](x);
    }

    del
}

/// Wolfe conditions
///
/// return true if Wolfe conditions are satisfied, otherwise false.
/// see also:
/// [Wolfe conditions - Wikipedia](https://en.wikipedia.org/wiki/Wolfe_conditions)
/// [Backtracking line search - Wikipedia](https://en.wikipedia.org/wiki/Backtracking_line_search)
///
/// * `f` - f(x).
/// * `del_f` - del(f)
/// * `p` - gradient descent vector.
/// * `alpha` - newton step size.
/// * `c1` - coefficient for condition 1.
/// * `c2` - coefficient for condition 2.
/// * `x_0` - start point.
/// * `x_1` - next point (x_0 + alpha * p).
pub fn is_satisfied_wolfe_conditions(
    f: FuncX,
    del_f: &Vec<&FuncX>,
    p: ArrayView1<f64>,
    alpha: f64,
    c1: f64,
    c2: f64,
    x_0: ArrayView1<f64>,
    x_1: ArrayView1<f64>,
) -> bool {
    // del_f0 is inner product (p , del(f(x_k))) < 0 because p ~= -del(f).
    let del_f0 = del(&del_f, x_0);
    let del_f1 = del(&del_f, x_1);
    println!("del_f0={}, del_f1={}", del_f0, del_f1);

    if f(x_1) <= f(x_0) + c1 * alpha * p.dot(&del_f0) && -p.dot(&del_f1) <= -c2 * p.dot(&del_f0) {
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assert::norm_l2;
    use core::f64;
    use ndarray::{arr1, ArrayView1};

    #[test]
    fn quasi_newton_1x1() {
        // f(x) = x^2 - 2x + 1
        let eps = 1e-6;
        let _x_0 = arr1(&[0.0]);
        let f: FuncX = |x: ArrayView1<f64>| x[0].powf(2.0) - 2.0 * x[0] + 1.0;
        let f_x: FuncX = |x: ArrayView1<f64>| 2.0 * x[0] - 2.0;
        let del_f = vec![&f_x];
        let x_0 = arr1(&[0.0]);
        let ans = arr1(&[1.0]);
        let x_k1 = bfgs(f, del_f, x_0.view());
        println!("x={:?}, truth={:?}", x_k1, ans);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }

    #[test]
    fn quasi_newton_2x2_quadratic() {
        // min f(x,y) = 2(x-1)^2 -2xy + (y-1)^2
        // minimum: (x,y) = (3,4)
        let f: FuncX = |x: ArrayView1<f64>| {
            2.0 * (x[0] - 1.0).powf(2.0) - 2.0 * x[0] * x[1] + (x[1] - 1.0).powf(2.0)
        };
        let f_x: FuncX = |x: ArrayView1<f64>| 4.0 * x[0] - 2.0 * x[1] - 4.0;
        let f_y: FuncX = |x: ArrayView1<f64>| -2.0 * x[0] + 2.0 * x[1] - 2.0;
        let del_f = vec![&f_x, &f_y];

        let eps = 1e-8;
        let x_0 = arr1(&[0.0, 0.0]);
        let ans = arr1(&[3.0, 4.0]);
        let x_k1 = bfgs(f, del_f, x_0.view());
        println!("x={:?}", x_k1);
        println!("truth={:?}", ans);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }

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
        let alpha = line_search(f, &del_f, p.view(), x_0.view());

        // assert f(x + alpha * p) < f(x)
        let x_1 = &x_0 + (Array1::from_elem(n, alpha) * &p);
        println!(
            "alpha={}, f(x0)={}, x1={}, f(x1)={}",
            alpha,
            f(x_0.view()),
            x_1,
            f(x_1.view())
        );
        assert!(f(x_1.view()) < f(x_0.view()));
    }

    #[test]
    fn line_search_2d() {
        // f(x) = 2x^2 -2xy + y^2
        let f = |x: ArrayView1<f64>| 2.0 * x[0].powf(2.0) - 2.0 * x[0] * x[1] + x[1].powf(2.0);
        let f_x: FuncX = |x: ArrayView1<f64>| 4.0 * x[0] - 2.0 * x[1];
        let f_y: FuncX = |x: ArrayView1<f64>| 2.0 * x[1] - 2.0 * x[0];
        let del_f = vec![&f_x, &f_y];
        let x_0 = arr1(&[-1.0, -1.0]);
        let p = arr1(&[0.5, 0.5]);
        let n = x_0.len();

        // calculate alpha (inexact line search)
        let alpha = line_search(f, &del_f, p.view(), x_0.view());

        // assert f(x + alpha * p) < f(x)
        let x_1 = &x_0 + (Array1::from_elem(n, alpha) * &p);
        println!(
            "alpha={}, f(x0)={}, x1={}, f(x1)={}",
            alpha,
            f(x_0.view()),
            x_1,
            f(x_1.view())
        );
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
