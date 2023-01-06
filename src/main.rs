mod assert;
mod gauss_seidel;
mod jacobi;
mod newton;
mod quasi_newton;

use ndarray::{arr1, Array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use newton::FuncX;

fn main() {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("dummy.csv")
        .unwrap();
    let n = rdr.records().count();
    println!("{}", n);

    // ラベルと特徴量に分ける
    let mut y = Array::zeros(n);
    let mut features = Array::zeros((n, 123));

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("dummy.csv")
        .unwrap();
    for (i, record) in rdr.records().enumerate() {
        let record = record.unwrap();
        y[i] = record[0].parse::<i8>().unwrap();
        for (j, data) in record.iter().enumerate() {
            if j > 0 {
                let index = data.parse::<usize>().unwrap() - 1;
                features[[i, index]] = 1.0;
            }
        }
    }

    println!("{:?}, {:?},{:?}", y, features, features.shape()[1]);
    println!("{:?}", &features.shape()[1]);
    let mut regression = Regression::new(y, features.clone());
    regression.train();
    //println!("loss={}", regression.loss(features.view()));

    let _eps = 1e-6;
    let _x_0 = arr1(&[0.0]);
    let _f: FuncX = |x: ArrayView1<f64>| x[0].powf(2.0) - 2.0 * x[0] + 1.0;
    let f_x: FuncX = |x: ArrayView1<f64>| 2.0 * x[0] - 2.0;
    let _del_f = vec![&f_x];
    let _x_0 = arr1(&[0.0]);
    let _ans = arr1(&[1.0]);
    //let x_k1 = bfgs(f, del_f, x_0.view());
    //println!("x={:?}, truth={:?}", x_k1, ans);
    //assert!(norm_l2((ans - x_k1).view()) < eps);
}

struct Regression {
    y: Array1<i8>,
    features: Array2<f64>,
    w: Array1<f64>,
    n: usize,
    l: usize,
    del_f: Array1<f64>,
}

impl Regression {
    fn new(y: Array1<i8>, features: Array2<f64>) -> Regression {
        let l = y.len();
        let n = features.shape()[1];
        let w = Array1::zeros(n);
        let del_f = Array1::zeros(n);
        Regression {
            y,
            features,
            w,
            n,
            l,
            del_f,
        }
    }

    fn loss(&self, w: ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;

        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                wx += w[j] * self.features[[i, j]];
            }
            let yi = f64::from(self.y[i]);
            sum += f64::ln(1.0 + f64::exp(-yi * wx));
        }

        sum
    }

    fn derivative(&self, w: ArrayView1<f64>, del_f: &mut ArrayViewMut1<f64>) {
        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                // todo: modify w as parameter.
                wx += w[j] * self.features[[i, j]];
            }

            let yi = f64::from(self.y[i]);

            for j in 0..self.n {
                del_f[j] += -yi * self.features[[i, j]] / (1.0 + f64::exp(yi * wx));
            }
        }
    }

    fn train(&mut self) {
        let delta = 0.1;
        let mut h: Array2<f64> = Array2::eye(self.n);
        let mut p: Array1<f64>;
        let mut s: Array1<f64>;
        let mut y: Array1<f64>;
        let mut w = Array1::zeros(self.n);
        let mut del_f_k_ = self.del_f.to_owned();
        let mut del_f_k1 = Array1::zeros(self.n);
        let g_norm = f64::abs(assert::norm_l2(del_f_k_.view()));
        self.derivative(w.view(), &mut del_f_k_.view_mut());

        while f64::abs(assert::norm_l2(del_f_k_.view())) > g_norm * delta {
            p = -&h.dot(&del_f_k_);
            let alpha = self.line_search(p.view(), w.view(), del_f_k_.view());
            if alpha == 0.0 {
                println!("w={:?}", w);
                break;
            }
            s = alpha * &p;
            w += &s;

            self.derivative(w.view(), &mut del_f_k1.view_mut());
            y = &del_f_k1 - &del_f_k_;

            let rho = 1.0 / &y.dot(&s);
            let lhm = Array2::eye(self.n) - rho * quasi_newton::x_yt(s.view(), y.view());
            let rhm = Array2::eye(self.n) - rho * quasi_newton::x_yt(y.view(), s.view());
            h = lhm.dot(&h.view()).dot(&rhm.view()) + rho * quasi_newton::x_yt(s.view(), s.view());

            Regression::copy(&mut del_f_k_.view_mut(), del_f_k1.view());
            //println!("{:?}", del_f_k_);
        }
    }

    fn copy(x: &mut ArrayViewMut1<f64>, y: ArrayView1<f64>) {
        let n = x.len();
        for i in 0..n {
            x[i] = y[i];
        }
    }

    fn line_search(
        &self,
        p: ArrayView1<f64>,
        x_0: ArrayView1<f64>,
        del_f0: ArrayView1<f64>,
    ) -> f64 {
        let rho = 0.5;
        let max_iteration = 100;
        let mut iteration = 0;
        let mut alpha = 1.0;
        let mut del_f1 = Array1::zeros(self.n);
        loop {
            let x_1 = alpha * &p + &x_0;
            self.derivative(x_1.view(), &mut del_f1.view_mut());
            if self.is_satisfied_wolfe_conditions(alpha, p, x_0, x_1.view(), del_f0, del_f1.view())
            {
                return alpha;
            }

            iteration += 1;
            if iteration >= max_iteration {
                return 0.0;
            }
            alpha *= rho;
        }
    }

    pub fn is_satisfied_wolfe_conditions(
        &self,
        alpha: f64,
        p: ArrayView1<f64>,
        w_0: ArrayView1<f64>,
        w_1: ArrayView1<f64>,
        del_f0: ArrayView1<f64>,
        del_f1: ArrayView1<f64>,
    ) -> bool {
        let c1 = 1e-4;
        let c2 = 0.9;
        if self.loss(w_1) <= self.loss(w_0) + c1 * alpha * p.dot(&del_f0)
            && -p.dot(&del_f1) <= -c2 * p.dot(&del_f0)
        {
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn new() {
        // if w is 0 vector, sum of loss becomes l * log(2).
        let l = 10;
        let n = 1;
        let y = Array::zeros(l);
        let features = Array::zeros((l, n));
        let regression = Regression::new(y, features.clone());
        let w = Array1::<f64>::zeros(n);
        assert_eq!(6.931471805599453, regression.loss(w.view()));
    }

    #[test]
    fn derivative() {
        // if w is 0 vector
        let n = 2;
        let y = arr1(&[-1, -1]);
        let features = arr2(&[[1.0, 0.0], [1.0, 0.0]]);
        let mut regression = Regression::new(y, features);
        let mut del_f = Array1::zeros(regression.n);
        let w = Array1::<f64>::zeros(n);

        regression.derivative(w.view(), &mut del_f.view_mut());

        assert_eq!(arr1(&[1.0, 0.0]), &del_f);
    }

    #[test]
    fn wolfe_conditions() {}
}
