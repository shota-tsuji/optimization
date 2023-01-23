use optimization::assert;
use optimization::linear_algebra as la;

use ndarray::{Array, Array1, Array2};
use std::process;

fn main() {
    // ラベルと特徴量に分ける
    let path = String::from("./dummy.csv");
    let (y, mat_x) = load_csv(path);
    // unique
    // index array for value == -1
    let y_nega_idx: Vec<usize> = y
        .iter()
        .enumerate()
        .filter(|(_, &val)| val == -1)
        .map(|(i, _)| i)
        .collect();
    // ones-array
    // zero with index-array
    let mut y_bin = Array1::ones(y.len());
    for i in y_nega_idx {
        y_bin[i] = 0;
    }

    let mut regression = Regression::new(y_bin, mat_x.clone());
    let w = Array1::zeros(regression.n);
    let mut g = Array1::zeros(regression.n);
    regression.derivative(&w, &mut g);

    println!("loss={}", regression.loss(&w));
    //println!("g={:?}", &g);
    println!("|g|={}", assert::norm_l2(&g));
    regression.train();
}

fn load_csv(path: String) -> (Array1<i8>, Array2<f64>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&path)
        .unwrap();
    let n = rdr.records().count();
    //println!("{}", n);
    let mut y = Array::zeros(n);
    let mut features = Array::zeros((n, 123));

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&path)
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

    (y, features)
}

struct Regression {
    y: Array1<i8>,
    features: Array2<f64>,
    // number of feature parameters
    n: usize,
    // number of train data
    l: usize,
    num_newton_step: i64,
    num_quasi_newton: i64,
}

impl Regression {
    fn new(y: Array1<i8>, features: Array2<f64>) -> Regression {
        let l = y.len();
        let n = features.shape()[1];
        let num_newton_step = 0;
        let num_quasi_newton = 0;
        Regression {
            y,
            features,
            n,
            l,
            num_newton_step,
            num_quasi_newton,
        }
    }

    // todo: modify loss
    fn loss(&self, w: &Array1<f64>) -> f64 {
        let mut sum = 0.0;

        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                wx += w[j] * self.features[[i, j]];
            }
            let yi = f64::from(self.y[i]);
            sum += f64::ln(1.0 + f64::exp(wx)) - yi * wx;
        }

        sum
    }

    // todo: modify gradient
    fn derivative(&self, w: &Array1<f64>, g: &mut Array1<f64>) {
        for i in 0..g.len() {
            g[i] = 0.0;
        }
        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                // todo: modify features as parameter.
                wx += w[j] * self.features[[i, j]];
            }

            let yi = f64::from(self.y[i]);

            for j in 0..self.n {
                let p = 1.0 / (1.0 + f64::exp(-wx));
                //g[j] += -yi * self.features[[i, j]] / (1.0 + f64::exp(yi * wx));
                g[j] += self.features[[i, j]] * (p - yi);
            }
        }
    }

    fn train(&mut self) {
        // const scala
        let delta = 1e-6;
        let n = self.n;

        // const vector
        let mut p: Array1<f64>;
        let mut s: Array1<f64>;
        let mut y: Array1<f64>;
        let mut w = Array1::zeros(n);
        let mut g = Array1::zeros(n);
        let mut g_new = Array1::zeros(n);
        self.derivative(&w, &mut g);

        // const matrix
        let mut h: Array2<f64> = Array2::eye(n);
        let mat_i = Array2::<f64>::eye(n);

        let g_norm = f64::abs(assert::norm_l2(&g));
        let mut i = 0;

        // check gradient is enough small.
        while f64::abs(assert::norm_l2(&g)) > g_norm * delta {
            self.num_quasi_newton += 1;
            p = -&h.dot(&g);
            let alpha = self.line_search(&p, &w, &g);
            if alpha == 0.0 {
                eprintln!("line search failed.");
                process::exit(1);
            }
            s = alpha * &p;
            w += &s;

            self.derivative(&w, &mut g_new);
            y = &g_new - &g;

            let rho = 1.0 / &y.dot(&s);
            let lhm = &mat_i - rho * la::x_yt(&s, &y);
            let rhm = &mat_i - rho * la::x_yt(&y, &s);
            h = lhm.dot(&h).dot(&rhm) + rho * la::x_yt(&s, &s);

            println!(
                "{}, residual: {:?}",
                self.num_quasi_newton,
                f64::abs(assert::norm_l2(&g))
            );
            la::copy(&mut g, &g_new);
            i += 1;
            if i >= 4 {
                println!("failed");
                break;
            }
        }
        println!("number of newton step: {}", self.num_newton_step);
        println!("number of quasi-newton: {}", self.num_quasi_newton);
        println!("initial residual of gradient: {}", g_norm);
        println!("residual of gradient: {}", assert::norm_l2(&g));
    }

    fn line_search(&mut self, p: &Array1<f64>, x_0: &Array1<f64>, del_f0: &Array1<f64>) -> f64 {
        let rho = 0.5;
        let max_iteration = 100;
        let mut iteration = 0;
        let mut alpha = 1.0;
        let mut del_f1 = Array1::zeros(self.n);
        loop {
            self.num_newton_step += 1;
            let x_1 = alpha * p + x_0;
            self.derivative(&x_1, &mut del_f1);
            if self.is_satisfied_wolfe_conditions(alpha, p, x_0, &x_1, del_f0, &del_f1) {
                return alpha;
            }

            //println!(
            //    "\tls_{},del_f={}",
            //    iteration,
            //    f64::abs(assert::norm_l2(del_f1.view()))
            //);
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
        p: &Array1<f64>,
        w_0: &Array1<f64>,
        w_1: &Array1<f64>,
        del_f0: &Array1<f64>,
        del_f1: &Array1<f64>,
    ) -> bool {
        let c1 = 1e-4;
        let c2 = 0.9;
        if self.loss(w_1) <= self.loss(w_0) + c1 * alpha * p.dot(del_f0)
            && p.dot(del_f1).abs() <= c2 * p.dot(del_f0).abs()
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
        assert_eq!(6.931471805599453, regression.loss(&w));
    }

    #[test]
    fn derivative() {
        // if w is 0 vector
        let n = 2;
        let y = arr1(&[-1, -1]);
        let features = arr2(&[[1.0, 0.0], [1.0, 0.0]]);
        let regression = Regression::new(y, features);
        let mut del_f = Array1::zeros(regression.n);
        let w = Array1::<f64>::zeros(n);

        regression.derivative(&w, &mut del_f);

        assert_eq!(arr1(&[3.0, 0.0]), &del_f);
    }

    #[test]
    fn wolfe_conditions() {}
}
