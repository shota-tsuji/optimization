use ndarray::{arr1, Array1, Array2, ArrayView1, Dim};

pub fn newton(
    x_0: ArrayView1<f64>,
    del_f: Vec<impl Fn(ArrayView1<f64>) -> f64>,
    h: Vec<Vec<impl Fn(ArrayView1<f64>) -> f64>>,
) -> Array1<f64> {
    let delta = 1e-10;
    let mut x_k_ = x_0.to_owned();
    let mut x_k1 = Array1::zeros(x_k_.raw_dim());
    let n = x_0.len();
    let mut a = Array2::zeros((n, n));
    let mut b = Array1::zeros((n));
    loop {
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = h[i][j](x_k_.view());
            }
        }
        println!("a={:?}", a);

        for i in 0..n {
            b[[i]] = -del_f[i](x_k_.view());
        }
        println!("b={:?}", b);

        x_k1[0] = x_k_[0] + b[[0]] / a[[0, 0]];
        if norm_l2((&x_k1 - &x_k_).view()) < delta {
            break;
        }
        x_k_[[0]] = x_k1[0];
        println!("x={}", x_k1)
    }

    x_k1
}

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
        let f_x = |x: ArrayView1<f64>| 2.0 * x[[0]] - 2.0;
        let del_f = vec![&f_x];
        let f_xx = |x: ArrayView1<f64>| 2.0;
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
        let f_x = |x: ArrayView1<f64>| 2.0 * x[[0]] - 4.0;
        let del_f = vec![&f_x];
        let f_xx = |x: ArrayView1<f64>| 2.0;
        let h = vec![vec![&f_xx]];
        let ans = arr1(&[2.0]);
        let x_k1 = newton(x_0.view(), del_f, h);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }

    #[test]
    fn newton_1x1_power3() {
        // f(x) = x^3 -2x^3 + x + 3
        let eps = 1e-10;
        let x_0 = arr1(&[2.0]);
        let f_x = |x: ArrayView1<f64>| 3.0 * x[[0]] * x[[0]] - 4.0 * x[[0]] + 1.0;
        let del_f = vec![&f_x];
        let f_xx = |x: ArrayView1<f64>| 6.0 * x[[0]] - 4.0;
        let h = vec![vec![&f_xx]];
        let ans = arr1(&[1.0]);
        let x_k1 = newton(x_0.view(), del_f, h);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }
}
