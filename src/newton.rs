use crate::gauss_seidel::gauss_seidel;
use crate::jacobi::jacobi;
use ndarray::{Array1, Array2, ArrayView1};

/// Newton method
///
/// returns local minimum of f(x).
///
/// * `x_0` - start point.
/// * `del_f` - del(f).
/// * `h` - Hessian matrix of f(x).
pub fn newton(x_0: ArrayView1<f64>, del_f: Vec<&FuncX>, h: Vec<Vec<&FuncX>>) -> Array1<f64> {
    let delta = 1e-10;
    let eps = 1e-10;
    let mut x_k_ = x_0.to_owned();
    let mut x_k1: Array1<f64>;
    let n = x_0.len();
    let mut a = Array2::zeros((n, n));
    let mut b = Array1::zeros(n);
    loop {
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = h[i][j](x_k_.view());
            }
        }

        for i in 0..n {
            b[i] = -del_f[i](x_k_.view());
        }

        // calculate delta-x and add it to current-x
        if cfg!(feature = "gauss-seidel") {
            x_k1 = &x_k_ + gauss_seidel(a.view(), Array1::zeros(n).view(), b.view(), eps);
        } else {
            x_k1 = &x_k_ + jacobi(a.view(), Array1::zeros(n).view(), b.view(), eps);
        }
        if norm_l2((&x_k1 - &x_k_).view()) < delta {
            break;
        }

        for i in 0..n {
            x_k_[i] = x_k1[i];
        }
    }

    x_k1
}

/// Math function type alias.
///
/// This type is used to `coercion` from function items to function pointers.
/// https://doc.rust-lang.org/beta/reference/types/function-item.html
/// https://stackoverflow.com/questions/27895946/expected-fn-item-found-a-different-fn-item-when-working-with-function-pointer
pub type FuncX = fn(ArrayView1<f64>) -> f64;

fn norm_l2(x: ArrayView1<f64>) -> f64 {
    x.map(|x| x * x).sum().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn newton_1x1() {
        // f(x) = x^2 - 2x + 1
        let eps = 1e-10;
        let x_0 = arr1(&[0.0]);
        let f_x: FuncX = |x: ArrayView1<f64>| 2.0 * x[0] - 2.0;
        let del_f = vec![&f_x];
        let f_xx: FuncX = |_x: ArrayView1<f64>| 2.0;
        let h = vec![vec![&f_xx]];
        let ans = arr1(&[1.0]);
        let x_k1 = newton(x_0.view(), del_f, h);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }

    #[test]
    fn newton_1x1_other() {
        // f(x) = x^2 -4x +4
        let eps = 1e-10;
        let x_0 = arr1(&[0.0]);
        let f_x: FuncX = |x: ArrayView1<f64>| 2.0 * x[0] - 4.0;
        let del_f = vec![&f_x];
        let f_xx: FuncX = |_x: ArrayView1<f64>| 2.0;
        let h = vec![vec![&f_xx]];
        let ans = arr1(&[2.0]);
        let x_k1 = newton(x_0.view(), del_f, h);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }

    #[test]
    fn newton_1x1_power3() {
        // min f(x) = x^3 - 2x^2 + x + 3
        // local minimum: x = 1.0
        let eps = 1e-10;
        let x_0 = arr1(&[2.0]);
        let f_x: FuncX = |x: ArrayView1<f64>| 3.0 * x[0] * x[0] - 4.0 * x[0] + 1.0;
        let del_f = vec![&f_x];
        let f_xx: FuncX = |x: ArrayView1<f64>| 6.0 * x[0] - 4.0;
        let h = vec![vec![&f_xx]];
        let ans = arr1(&[1.0]);
        let x_k1 = newton(x_0.view(), del_f, h);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }

    #[test]
    fn newton_2x2_quadratic() {
        // min f(x,y) = 2(x-1)^2 -2xy + (y-1)^2
        // minimum: (x,y) = (3,4)
        let eps = 1e-8;
        let x_0 = arr1(&[0.0, 0.0]);

        let f_x: FuncX = |x: ArrayView1<f64>| 4.0 * x[0] - 2.0 * x[1] - 4.0;
        let f_y: FuncX = |x: ArrayView1<f64>| -2.0 * x[0] + 2.0 * x[1] - 2.0;
        let del_f = vec![&f_x, &f_y];

        let f_xx: FuncX = |_x: ArrayView1<f64>| 4.0;
        let f_xy: FuncX = |_x: ArrayView1<f64>| -2.0;
        let f_yy: FuncX = |_x: ArrayView1<f64>| 2.0;
        let h = vec![vec![&f_xx, &f_xy], vec![&f_xy, &f_yy]];

        let ans = arr1(&[3.0, 4.0]);
        let x_k1 = newton(x_0.view(), del_f, h);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }
}
