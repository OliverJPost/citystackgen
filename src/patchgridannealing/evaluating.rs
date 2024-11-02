use std::cmp::max;
use std::fs::File;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use bytemuck::cast_slice;
use image::codecs::tiff::TiffEncoder;
use image::{ColorType, ImageEncoder};
use ndarray::Array2;
//use rerun::{RecordingStream, Scalar};
use crate::patchgridannealing::Aggregator;
use crate::patchgridannealing::metrics::{PatchMetric, PatchGridMetricID};
use crate::patchgridannealing::patchgrid::PatchGrid;
use crate::patchgridannealing::statistics::PatchGridStatistics;
use rayon::prelude::*;
use rerun::{Tensor, TensorData};
use tiff::encoder::colortype::Gray32;
use crate::stream::{ChunkedRerunStream, Stream}; // Import rayon's parallel iterator trait
type RecordingStream = ChunkedRerunStream;

pub trait Evaluator {
    fn evaluate(&mut self, grid: &mut PatchGrid) -> f64;
    fn print_progress(&self, grid: &PatchGrid, rec: Option<Rc<RecordingStream>>);
}

pub enum Comparer {
    CosineSimilarity,
    Intersection
}

impl Comparer {
    fn compare(&self, a: &Vec<f64>, b: &Vec<f64>) -> Result<f64, &'static str> {
        match self {
            Comparer::CosineSimilarity => cosine_similarity(a, b),
            Comparer::Intersection => intersection(a, b)
        }
    }
}


fn intersection(a: &Vec<f64>, b: &Vec<f64>) -> Result<f64, &'static str> {
    let mut intersection = 0.0;
    let maximum_possible_intersection: f64 = if a.iter().sum::<f64>() > b.iter().sum::<f64>() {
        a.iter().sum()
    } else {
        b.iter().sum()
    };
    for (ai, bi) in a.iter().zip(b.iter()) {
        intersection += ai.min(*bi);
    }
    let intersection = intersection / maximum_possible_intersection;
    if intersection.is_nan() {
        return Err("Intersection is NaN");
    }
    Ok(intersection)
}

fn cosine_similarity(a: &Vec<f64>, b: &Vec<f64>) -> Result<f64, &'static str> {
    if a.is_empty() || b.is_empty() {
        return Err("Vectors must not be empty");
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (ai, bi) in a.iter().zip(b.iter()) {
        dot_product += ai * bi;
        norm_a += ai.powi(2);
        norm_b += bi.powi(2);
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return Err("Vectors must not be zero vectors");
    }

    let similarity = dot_product / (norm_a.sqrt() * norm_b.sqrt());
    Ok(similarity)
}

pub struct GridStatisticsComparer {
    target_statistics: PatchGridStatistics,
    comparisons: Vec<(PatchGridMetricID, PatchGridMetricID, Aggregator)>,
    comparer: Comparer,
    old_similarities: Vec<f64>
}

impl GridStatisticsComparer{
    pub fn from_target(target_statistics: PatchGridStatistics, comparer: Comparer) -> Self {
        GridStatisticsComparer {
            target_statistics,
            comparisons: vec![],
            comparer,
            old_similarities: vec![]
        }
    }

    pub fn with_binned_comparison(mut self, metric: PatchGridMetricID, bin_metric: PatchGridMetricID, aggregator: Aggregator) -> Self {
        self.comparisons.push((metric, bin_metric, aggregator));
        self
    }

    fn compare(&mut self, grid: &mut PatchGrid) -> f64 {
        let mut total_similarity = 0.0;
        // TODO multithreaded
        let mut n_comparisons = 0;
        let modified_classes = grid.drain_affected_class_ids();
        for class in grid.class_ids() {
            // if class.is_negative() || class == 99 {
            //     // Skip None classes
            //     continue;
            // }
            if !modified_classes.contains(&class) {
                let re = self.old_similarities.get(class as usize);
                if let Some(old_similarity) = re {
                    total_similarity += old_similarity;
                    n_comparisons += self.comparisons.len();
                    continue;
                }
            }

            let grid_patch_indices = grid.statistics.get_patch_indices(class);
            let target_patch_indices = self.target_statistics.get_patch_indices(class);
            let mut class_score = 0.0;
            let areas = grid.statistics.patch_metrics.get(&PatchGridMetricID::PatchArea).unwrap().values(&grid_patch_indices);
            let target_areas = self.target_statistics.patch_metrics.get(&PatchGridMetricID::PatchArea).unwrap().values(&target_patch_indices);
            for (metric, bin_metric, aggregator) in &self.comparisons {
                let target_metric = self.target_statistics.patch_metrics.get(metric).unwrap().values(&target_patch_indices);
                let target_bin_metric = self.target_statistics.patch_metrics.get(bin_metric).unwrap().values(&target_patch_indices);
                let grid_metric = grid.statistics.patch_metrics.get(metric).unwrap().values(&grid_patch_indices);
                let grid_bin_metric = grid.statistics.patch_metrics.get(bin_metric).unwrap().values(&grid_patch_indices);
                // FIXME hardcoded step and max dist
                // let max_dist_target = *target_bin_metric.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                // let max_dist_grid = *grid_bin_metric.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                // let max_dist = max_dist_target.max(max_dist_grid);
                // let step = max_dist / 5.;
                let max_dist = 50_000.; //FIXME hardcoded
                let step = 500.;
                let grid_bins = aggregator.aggregate(&grid_metric, &grid_bin_metric, step, max_dist);
                let target_bins = aggregator.aggregate(&target_metric, &target_bin_metric,  step, max_dist);
                let score = self.comparer.compare(&grid_bins, &target_bins).unwrap_or_else(|a| {
                    1.0
                }
                );
                total_similarity += score;
                class_score += score;

                n_comparisons += 1;
            }
            if self.old_similarities.len() <= class as usize {
                self.old_similarities.resize(class as usize + 1, 0.0);
            }
            self.old_similarities[class as usize] = class_score;

        }
        if n_comparisons == 0 {
            panic!("No comparisons made");
        }
        let score = total_similarity / n_comparisons as f64;
        if score.is_nan() {
            panic!("Score is NaN");
        };
        if score == 0.0 {
            panic!("Score is zero");
        }
        score
    }

    fn compare_multithreaded(&self, grid: &PatchGrid) -> f64 {
        let chunk_size = 3; // Example chunk size
        let similarities: Vec<f64> = grid.class_ids()
            .par_chunks(chunk_size)
            .flat_map(|class_chunk| {
                class_chunk.iter().map(|class| {
                    if class.is_negative() || *class == 99 {
                        return 0.0;
                    }

                    let grid_patch_indices = grid.statistics.get_patch_indices(*class);
                    let target_patch_indices = self.target_statistics.get_patch_indices(*class);
                    let mut local_similarity = 0.0;
                    let areas = grid.statistics.patch_metrics.get(&PatchGridMetricID::PatchArea).unwrap().values(&grid_patch_indices);
                    let target_areas = self.target_statistics.patch_metrics.get(&PatchGridMetricID::PatchArea).unwrap().values(&target_patch_indices);

                    for (metric, bin_metric, aggregator) in &self.comparisons {
                        let target_metric = self.target_statistics.patch_metrics.get(metric).unwrap().values(&target_patch_indices);
                        let target_bin_metric = self.target_statistics.patch_metrics.get(bin_metric).unwrap().values(&target_patch_indices);
                        let grid_metric = grid.statistics.patch_metrics.get(metric).unwrap().values(&grid_patch_indices);
                        let grid_bin_metric = grid.statistics.patch_metrics.get(bin_metric).unwrap().values(&grid_patch_indices);

                        let max_dist = 50_000.;
                        let step = 500.;

                        let grid_bins = aggregator.aggregate(&grid_metric, &grid_bin_metric, step, max_dist);
                        let target_bins = aggregator.aggregate(&target_metric, &target_bin_metric, step, max_dist);

                        local_similarity += self.comparer.compare(&grid_bins, &target_bins).unwrap_or(0.0);
                    }

                    local_similarity
                }).collect::<Vec<_>>()
            })
            .collect();

        // Sum up all the local similarities to get the total similarity
        let n_comparisons_value: usize = similarities.iter().count() * self.comparisons.len();
        let total_similarity_value: f64 = similarities.into_iter().sum();

        if n_comparisons_value == 0 {
            panic!("No comparisons made");
        }
        let score = total_similarity_value / n_comparisons_value as f64;
        if score.is_nan() {
            panic!("Score is NaN");
        };
        if score == 0.0 {
            panic!("Score is zero");
        }
        score
    }

    fn compare_recorded(&self, grid: &PatchGrid, rec: &Option<Rc<RecordingStream>>) -> f64 {
        let mut total_similarity = 0.0;
        // TODO multithreaded
        let mut n_comparisons = 0;
        for class in grid.class_ids() {
            // if class.is_negative() || class == 99 {
            //     // Skip None classes
            //     continue;
            // }
            let grid_patch_indices = grid.statistics.get_patch_indices(class);
            let target_patch_indices = self.target_statistics.get_patch_indices(class);
            let areas = grid.statistics.patch_metrics.get(&PatchGridMetricID::PatchArea).unwrap().values(&grid_patch_indices);
            let target_areas = self.target_statistics.patch_metrics.get(&PatchGridMetricID::PatchArea).unwrap().values(&target_patch_indices);
            for (metric, bin_metric, aggregator) in &self.comparisons {
                let target_metric = self.target_statistics.patch_metrics.get(metric).unwrap().values(&target_patch_indices);
                let target_bin_metric = self.target_statistics.patch_metrics.get(bin_metric).unwrap().values(&target_patch_indices);
                let grid_metric = grid.statistics.patch_metrics.get(metric).unwrap().values(&grid_patch_indices);
                let grid_bin_metric = grid.statistics.patch_metrics.get(bin_metric).unwrap().values(&grid_patch_indices);
                // FIXME hardcoded step and max dist
                // let max_dist_target = *target_bin_metric.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                // let max_dist_grid = *grid_bin_metric.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                // let max_dist = max_dist_target.max(max_dist_grid);
                // let step = max_dist / 5.;
                let max_dist = 50_000.; //FIXME hardcoded
                let step = 500.;
                let grid_bins = aggregator.aggregate(&grid_metric, &grid_bin_metric, step, max_dist);
                let target_bins = aggregator.aggregate(&target_metric, &target_bin_metric, step, max_dist);
                let similarity = self.comparer.compare(&grid_bins, &target_bins).unwrap_or_else(|a| {
                    println!("Error comparing metrics {:?} and {:?} using {:?}: {:?} for class {:?}", metric, bin_metric, aggregator, a, class);
                    1.0
                }
                );
                total_similarity += similarity;
                rec.stream_scalar(&format!("metrics/class{:?}/{:?}/{:?}", class, metric, aggregator), similarity);
                n_comparisons += 1;
            }
        }
        if n_comparisons == 0 {
            panic!("No comparisons made");
        }
        let score = total_similarity / n_comparisons as f64;
        if score.is_nan() {
            panic!("Score is NaN");
        };
        if score == 0.0 {
            panic!("Score is zero");
        }
        score
    }
}

impl Evaluator for GridStatisticsComparer {
    fn evaluate(&mut self, grid: &mut PatchGrid) -> f64 {
        1./self.compare(grid) * 10.
    }

    fn print_progress(&self, grid: &PatchGrid, rec: Option<Rc<RecordingStream>>) {
        println!("Score: {:.3}", (1./self.compare_recorded(grid, &rec)) * 1000.);
        let max_patch_index = grid.statistics.patch_classes.len();
        let all_patch_idx = (0..max_patch_index).collect();
        // For each metric, make a grid and rec it
        let mut unique_metrics = vec![];
        for (metric, bin_metric, _) in &self.comparisons {
            if !unique_metrics.contains(metric) {
                unique_metrics.push(metric.clone());
            }
            if !unique_metrics.contains(bin_metric) {
                unique_metrics.push(bin_metric.clone());
            }
        }
        for metric in unique_metrics {
            let metric_values = grid.statistics.patch_metrics.get(&metric).unwrap().values(&all_patch_idx);
            let mut value_grid = Array2::zeros((grid.class_grid.shape()[0], grid.class_grid.shape()[1]));
            for (idx, value) in all_patch_idx.iter().zip(metric_values.iter()) {
                let patch_class = grid.statistics.patch_classes[*idx];
                if patch_class.is_none() || patch_class.unwrap().is_negative() || patch_class.unwrap() == 99 {
                    continue;
                }
                let patch = grid.statistics.patch_ranges[*idx].unwrap();
                for row in patch.0 .0..patch.0 .1 {
                    for col in patch.1 .0..patch.1 .1 {
                        if grid.patch_grid[[row, col]] == *idx {
                            value_grid[[row, col]] = *value;
                        }
                    }
                }
            }

            if let Some(rec) = &rec {
                let tensor= TensorData::try_from(value_grid).unwrap();
                rec.stream.log(format!("metrics/{:?}", metric), &Tensor::new(tensor));
            }
        }

    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use super::*;
    use crate::patchgridannealing::aggregating::Aggregator;
    use crate::patchgridannealing::metrics::{PatchArea, PatchDistanceToCenter};
    use crate::patchgridannealing::statistics::PatchGridStatistics;
    use ndarray::array;

    #[test]
    fn test_intersection() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 1.0];
        assert_eq!(intersection(&a, &b).unwrap(), 4.0/6.0);
    }



}