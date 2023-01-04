use ndarray::{arr1, Array, Array1, Array2, ArrayBase, ArrayView1};
use optimization::assert::norm_l2;
use optimization::newton::FuncX;
use optimization::quasi_newton::bfgs;

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
    println!("{:?}", features.shape()[1]);
    let regression = Regression::new(y, features);
    println!("loss={}", regression.loss());

    let eps = 1e-6;
    let _x_0 = arr1(&[0.0]);
    let f: FuncX = |x: ArrayView1<f64>| x[0].powf(2.0) - 2.0 * x[0] + 1.0;
    let f_x: FuncX = |x: ArrayView1<f64>| 2.0 * x[0] - 2.0;
    let del_f = vec![&f_x];
    let x_0 = arr1(&[0.0]);
    let ans = arr1(&[1.0]);
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

    fn loss(&self) -> f64 {
        let mut sum = 0.0;

        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                wx += self.w[j] * self.features[[i, j]];
            }
            let yi = f64::from(self.y[i]);
            sum += f64::ln(1.0 + f64::exp(-yi * wx));
        }

        sum
    }

    fn derivative(&mut self) {
        for i in 0..self.l {
            let mut wx = 0.0;
            for j in 0..self.n {
                wx += self.w[j] * self.features[[i, j]];
            }

            let yi = f64::from(self.y[i]);

            for j in 0..self.n {
                self.del_f[j] += -yi * self.features[[i, j]] / (1.0 + f64::exp(yi * wx));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::f64;
    use ndarray::{arr1, arr2, ArrayView1};

    #[test]
    fn new() {
        let n = 10;
        let mut y = Array::zeros(n);
        let mut features = Array::zeros((n, 1));
        let regression = Regression::new(y, features);
        assert_eq!(6.931471805599453, regression.loss());
    }

    #[test]
    fn derivative() {
        let n = 2;
        let mut y = arr1(&[-1, -1]);
        let mut features = arr2(&[[1.0, 0.0], [1.0, 0.0]]);
        let mut regression = Regression::new(y, features);

        regression.derivative();

        assert_eq!(arr1(&[1.0, 0.0]), regression.del_f);
    }
}
