use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{Error, Executor};
use argmin::solver::linesearch::HagerZhangLineSearch;
use argmin::solver::quasinewton::BFGS;

use ndarray::{Array1, Array2};

use optimization::regression::logistic as lg;
use optimization::utils;

fn main() {
    // divide to labels and samples
    let path = String::from("./dummy.csv");
    let (y, mat_x) = utils::load_csv(path);
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

    let logistic = lg::Logistic::new(y_bin, mat_x.clone());
    let mut regression = Regression {};
    regression.train(logistic);
}

struct Regression {}

impl Regression {
    fn train(&mut self, logistic: lg::Logistic) -> Result<(), Error> {
        let init_param = Array1::<f64>::zeros(logistic.n);
        let init_hessian = Array2::<f64>::eye(logistic.n);

        let linesearch = HagerZhangLineSearch::new();
        let solver = BFGS::new(linesearch);

        let res = Executor::new(logistic, solver)
            .configure(|state| {
                state
                    .param(init_param)
                    .inv_hessian(init_hessian)
                    .max_iters(60)
            })
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()?;

        std::thread::sleep(std::time::Duration::from_secs(1));

        println!("{res}");
        Ok(())
    }
}
