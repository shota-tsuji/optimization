use ndarray::{arr1, Array1, ArrayView1};

pub fn newton(
    x_0: ArrayView1<f64>,
    del_f: Vec<Box<impl Fn(f64) -> f64>>,
    f_xx: fn(f64) -> f64,
) -> Array1<f64> {
    let eps = 1e-10;
    let mut x_k_ = x_0.to_owned();
    let mut x_k1 = 0.0;
    loop {
        x_k1 = x_k_[0] - (*del_f[0])(x_k_[0]) / f_xx(x_k_[0]);
        if (x_k1 - x_k_[0]).abs() < eps {
            break;
        }
        x_k_[[0]] = x_k1;
        println!("{}", x_k1)
    }

    arr1(&[x_k1])
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
        let f_x = |x: f64| 2.0 * x - 2.0;
        let del_f = vec![Box::new(f_x)];
        let f_xx = |x: f64| 2.0;
        let h = vec![vec![f_xx]];
        let ans = arr1(&[1.0]);
        let x_k1 = newton(x_0.view(), del_f, f_xx);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }

    #[test]
    fn newton_1x1_other() {
        // f(x) = x^2 -4x +4
        let eps = 1e-10;
        let x_0 = arr1(&[0.0]);
        let f_x = |x: f64| 2.0 * x - 4.0;
        let del_f = vec![Box::new(f_x)];
        let f_xx = |x: f64| 2.0;
        let ans = arr1(&[2.0]);
        let x_k1 = newton(x_0.view(), del_f, f_xx);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }

    #[test]
    fn newton_1x1_power3() {
        // f(x) = x^3 -2x^3 + x + 3
        let eps = 1e-10;
        let x_0 = arr1(&[2.0]);
        let f_x = |x: f64| 3.0 * x * x + -4.0 * x + 1.0;
        let del_f = vec![Box::new(f_x)];
        let f_xx = |x: f64| 6.0 * x - 4.0;
        let ans = arr1(&[1.0]);
        let x_k1 = newton(x_0.view(), del_f, f_xx);
        assert!(norm_l2((ans - x_k1).view()) < eps);
    }

    fn norm_l2(x: ArrayView1<f64>) -> f64 {
        x.map(|x| x * x).sum().sqrt()
    }
}
