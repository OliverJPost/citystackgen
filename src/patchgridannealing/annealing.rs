use std::rc::Rc;
use rand::prelude::ThreadRng;
use rand::Rng;
use rand_distr::num_traits::real::Real;
use crate::patchgridannealing::evaluating::Evaluator;
use crate::patchgridannealing::patchgrid::PatchGrid;
use crate::stream::{ChunkedRerunStream, Stream};

type RecordingStream = ChunkedRerunStream;

pub enum CoolingSchedule {
    Exponential{cooling_rate: f64}
}

impl CoolingSchedule {
    fn cool(&self, temperature: f64, iteration: usize) -> f64 {
        match self {
            CoolingSchedule::Exponential{cooling_rate} => temperature * cooling_rate
        }
    }
}

pub enum Acceptor {
    MetrolopolisCriteria
}

impl Acceptor {
    fn is_accepted(&self, delta: f64, temperature: f64, mut rng: &mut ThreadRng) -> bool {
        match self {
            Acceptor::MetrolopolisCriteria => {
                let probability = (-delta / temperature).exp();
                let random: f64 = rng.gen();
                random < probability
            }
        }
    }
}


pub struct PatchGridAnnealer {
    grid: PatchGrid,
    evaluator: Box<dyn Evaluator>, // TODO ensure grid statistics and evaluator are compatible
    acceptor: Acceptor,
    cooler: CoolingSchedule,
    start_temperature: f64,
    final_temperature: f64,
    rec: Option<Rc<RecordingStream>>
}

impl PatchGridAnnealer {
    pub fn new(grid: PatchGrid, evaluator: Box<dyn Evaluator>, acceptor: Acceptor, cooler: CoolingSchedule, start_temperature: f64, final_temperature: f64, rec: Rc<RecordingStream>) -> Self {
        PatchGridAnnealer {
            grid,
            evaluator,
            acceptor,
            cooler,
            start_temperature,
            final_temperature,
            rec: Some(rec)
        }
    }

    pub fn anneal(mut self, mut rng: &mut ThreadRng) -> PatchGrid {
        let mut temperature = self.start_temperature;
        let mut energy = self.evaluator.evaluate(&mut self.grid);
        let mut i = 0;
        let mut cumulative_delta = 0.0;
        let mut rejected = 0;
        let mut time = std::time::Instant::now();
        while temperature > self.final_temperature {
            self.grid.random_swap(&mut rng);
            // if rng.gen::<f64>() < 0.5 {
            //     self.grid.random_swap(&mut rng);
            // } else {
            //     self.grid.random_walk(&mut rng);
            // }
            let new_energy = self.evaluator.evaluate(&mut self.grid);
            let delta = new_energy - energy;

            if self.acceptor.is_accepted(delta, temperature, &mut rng) {
                energy = new_energy;
                cumulative_delta += delta;
            } else {
                self.grid.revert_state();
                rejected += 1;
            }
            temperature = self.cooler.cool(temperature, i);
            if i % 5000 == 0 {
                println!("Temperature: {:.3}. Cumulative Delta: {:.3}. Rejected {}/5000", temperature, cumulative_delta, rejected);
                self.rec.stream_scalar("progress/temperature", temperature);
                self.rec.stream_scalar("progress/cumulative_delta", cumulative_delta);
                self.rec.stream_scalar("progress/rejected", rejected as f64);
                self.rec.stream_scalar("progress/energy", energy);
                self.rec.stream_scalar("tick", i as f64);
                self.evaluator.print_progress(&self.grid, self.rec.clone());
                println!("Iterations per second: {}", 5000. as f64 / time.elapsed().as_secs_f64());
                time = std::time::Instant::now();

                cumulative_delta = 0.0;
                rejected = 0;
                self.rec.stream_raster("image", self.grid.class_grid.clone());
            }
            i+=1;
        }
        self.grid
    }
}

#[cfg(test)]
mod tests {
    use crate::patchgridannealing;
    use crate::patchgridannealing::{Aggregator, PatchGridMetricID};
    use super::*;

    #[test]
    fn benchmark() {
        let temperature = 0.0011;
        let mut array = ndarray::Array2::<i32>::zeros((250, 300));
        let mut rng = rand::thread_rng();
        array.mapv_inplace(|_| rng.gen_range(0..10));
        let city_center = (10000., 20500.);

        let statistics = patchgridannealing::PatchGridStatistics::new()
            .with_metric(Box::new(patchgridannealing::metrics::PatchArea::new()))
            .with_metric(Box::new(patchgridannealing::metrics::PatchDistanceToCenter::new(city_center, 100.0)))
            .with_metric(Box::new(patchgridannealing::metrics::PatchCoreAreaIndex::new()))
            .with_metric(Box::new(patchgridannealing::metrics::PatchShapeIndex::new()))
            ;

        let array_clone = array.clone();
        let grid = patchgridannealing::PatchGrid::init(array_clone, statistics);
        println!("Unique values {:?}", grid.class_ids());

        let statistics = patchgridannealing::PatchGridStatistics::new()
            .with_metric(Box::new(patchgridannealing::metrics::PatchArea::new()))
            .with_metric(Box::new(patchgridannealing::metrics::PatchDistanceToCenter::new(city_center, 100.0)))
            .with_metric(Box::new(patchgridannealing::metrics::PatchCoreAreaIndex::new()))
            .with_metric(Box::new(patchgridannealing::metrics::PatchShapeIndex::new()))
            ;

        let copy_grid = patchgridannealing::PatchGrid::init(array, statistics);
        let target = copy_grid.statistics;
        let acceptor = patchgridannealing::Acceptor::MetrolopolisCriteria;
        let cooler = patchgridannealing::CoolingSchedule::Exponential { cooling_rate: 0.99999 };

        let evaluator = patchgridannealing::GridStatisticsComparer::from_target(target, patchgridannealing::Comparer::Intersection)
            .with_binned_comparison(PatchGridMetricID::PatchArea, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Average)
            .with_binned_comparison(PatchGridMetricID::PatchArea, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Sum)
            .with_binned_comparison(PatchGridMetricID::PatchArea, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Count)
            //.with_binned_comparison(PatchGridMetricID::PatchArea, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Std)
            .with_binned_comparison(PatchGridMetricID::PatchCoreAreaIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Average)
            //.with_binned_comparison(PatchGridMetricID::PatchCoreAreaIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Std)
            .with_binned_comparison(PatchGridMetricID::PatchShapeIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Average)
            //.with_binned_comparison(PatchGridMetricID::PatchShapeIndex, PatchGridMetricID::PatchDistanceToCenter, Aggregator::Std)
            .with_binned_comparison(PatchGridMetricID::PatchCoreAreaIndex, PatchGridMetricID::PatchArea, Aggregator::Average)
            ;

        let rec = Rc::new(ChunkedRerunStream::new(rerun::RecordingStreamBuilder::new("annealing_benchmark")
            .spawn().unwrap(), 4000, false));
        let annealer = patchgridannealing::PatchGridAnnealer::new(grid, Box::new(evaluator), acceptor, cooler, temperature, 0.001, rec);
        let mut rng = rand::thread_rng();
        let new_grid = annealer.anneal(&mut rng);
    }
}