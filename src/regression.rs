pub mod logistic;

use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{Error, Executor};
use argmin::solver::linesearch::HagerZhangLineSearch;
use argmin::solver::quasinewton::BFGS;
use ndarray::{Array1, Array2};

use logistic::Logistic;

pub struct Regression {}

impl Regression {
    pub fn train(&mut self, logistic: Logistic) -> Result<(), Error> {
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
