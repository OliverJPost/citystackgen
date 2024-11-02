use std::collections::HashMap;
use std::ops::Range;
use ndarray::Array2;
use crate::patchgridannealing::metrics::{PatchMetric, PatchGridMetricID};

pub struct PatchGridStatistics {
    pub(crate) patch_classes: Vec<Option<i32>>,
    pub(crate) patch_ranges: Vec<Option<((usize, usize), (usize, usize))>>,
    pub(crate) patch_metrics: HashMap<PatchGridMetricID, Box<dyn PatchMetric>>,
}

impl PatchGridStatistics {

    pub fn new() -> Self {
        PatchGridStatistics {
            patch_classes: vec![],
            patch_ranges: vec![],
            patch_metrics: HashMap::new()
        }
    }

    pub fn with_metric(mut self, metric: Box<dyn PatchMetric>) -> Self {
        self.patch_metrics.insert(metric.id(), metric);
        self
    }

    pub fn update_patches(&mut self, patch_ids: &Vec<usize>, patch_grid: &Array2<usize>, class_grid: &Array2<i32> ) {
        self.update_ranges(patch_ids, patch_grid);
        self.update_classes(patch_ids, patch_grid, class_grid);

        for metric in &mut self.patch_metrics {
            metric.1.update(patch_ids, patch_grid, &self.patch_ranges);
        }
    }

    pub fn get_patch_indices(&self, class: i32) -> Vec<usize> {
        self.patch_classes.iter().enumerate().filter(|(_, class_id)| class_id == &&Some(class)).map(|(idx, _)| idx).collect()
    }

    fn update_ranges(&mut self, patch_ids: &Vec<usize>, patch_grid: &Array2<usize>) {

        let max_patch_id = patch_ids.iter().max().unwrap();
        if self.patch_ranges.len() <= *max_patch_id {
            self.patch_ranges.resize(max_patch_id + 1, None);
        }

        for ((row, col), patch_id) in patch_grid.indexed_iter() {
            if patch_ids.contains(patch_id) {
                let patch_range = self.patch_ranges[*patch_id].unwrap_or(((row, row), (col, col)));
                let ((min_row, max_row), (min_col, max_col)) = patch_range;
                let new_range = (
                    (min_row.min(row), max_row.max(row)),
                    (min_col.min(col), max_col.max(col))
                );
                self.patch_ranges[*patch_id] = Some(new_range);
            }
        }
    }

    //TODO optimize
    fn update_classes(&mut self, patch_ids: &Vec<usize>, patch_grid: &Array2<usize>, class_grid: &Array2<i32>) {
        let mut patch_class_ids: HashMap<usize, Option<i32>> = HashMap::new();
        for patch_id in patch_ids {
            let mut class_id = None;
            let re = self.patch_ranges[*patch_id];
            if re.is_none() {
                patch_class_ids.insert(*patch_id, None);
                continue;
            }
            let range = re.unwrap();

            let ((min_row, max_row), (min_col, max_col)) = range;

            for row in min_row..=max_row {
                for col in min_col..=max_col {
                    if patch_grid[[row, col]] == *patch_id {
                        class_id = Some(class_grid[[row, col]]);
                        break;
                    }
                }
            }

            patch_class_ids.insert(*patch_id, class_id);
        }

        let max_patch_id = patch_class_ids.keys().max().unwrap();
        // Ensure patch_classes vector is long enough
        if self.patch_classes.len() <= *max_patch_id {
            self.patch_classes.resize(max_patch_id + 1, None);
        }

        for (patch_id, class_id) in patch_class_ids {
            self.patch_classes[patch_id] = class_id;
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use super::*;

    #[test]
    fn test_update_classes() {
        let mut class_grid = get_contiguous_class_grid();
        let mut patch_grid = get_contiguous_patch_grid();
        let mut stats = PatchGridStatistics::new();

        stats.update_ranges(&(0..=8).collect(), &patch_grid);
        stats.update_classes(&(0..=8).collect(), &patch_grid, &class_grid);
        assert_eq!(stats.patch_classes, vec![Some(0), Some(1), Some(2), Some(4), Some(1), Some(4), Some(1), Some(2), Some(3)]);

        // Changing class of one patch
        class_grid[(1, 1)] = 3;
        stats.update_ranges(&(0..=8).collect(), &patch_grid);
        stats.update_classes(&(0..=8).collect(), &patch_grid, &class_grid);
        assert_eq!(stats.patch_classes, vec![Some(0), Some(3), Some(2), Some(4), Some(1), Some(4), Some(1), Some(2), Some(3)]);

        // Dropping a patch
        class_grid[(1, 1)] = 2;
        patch_grid[(1, 1)] = 2;
        stats.update_ranges(&(0..=8).collect(), &patch_grid);
        stats.update_classes(&(0..=8).collect(), &patch_grid, &class_grid);
        assert_eq!(stats.patch_classes, vec![Some(0), None, Some(2), Some(4), Some(1), Some(4), Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn test_update_classes_noncontiguous() {
        let class_grid = get_noncontiguous_class_grid();
        let patch_grid = get_noncontiguous_patch_grid();
        let mut stats = PatchGridStatistics::new();

        stats.update_ranges(&(0..=12).collect(), &patch_grid);
        stats.update_classes(&(0..=12).collect(), &patch_grid, &class_grid);
        assert_eq!(stats.patch_classes, vec![Some(0), Some(1), None, Some(4), Some(1), Some(4), Some(1), None, Some(7), None, None, Some(2), Some(2)]);
    }

    #[test]
    fn test_get_patch_indices() {
        let class_grid = get_noncontiguous_class_grid();
        let patch_grid = get_noncontiguous_patch_grid();
        let mut stats = PatchGridStatistics::new();
        stats.update_ranges(&(0..=12).collect(), &patch_grid);
        stats.update_classes(&(0..=12).collect(), &patch_grid, &class_grid);

        assert_eq!(stats.get_patch_indices(1), vec![1, 4, 6]);
        assert_eq!(stats.get_patch_indices(7), vec![8]);
        let empty_vec: Vec<usize> = vec![];
        assert_eq!(stats.get_patch_indices(6), empty_vec);
    }

    fn get_contiguous_class_grid() -> Array2<i32>{
        array![
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 2, 2, 2, 0],
            [0, 4, 2, 2, 2, 2, 0],
            [0, 1, 4, 2, 2, 1, 0],
            [0, 2, 4, 4, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    }

    fn get_noncontiguous_class_grid() -> Array2<i32>{
        array![
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 2, 2, 2, 0],
            [0, 4, 2, 2, 2, 2, 0],
            [0, 1, 4, 2, 2, 1, 0],
            [0, 2, 4, 4, 7, 7, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    }

    fn get_contiguous_patch_grid() -> Array2<usize>{
        array![
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 2, 2, 2, 0],
            [0, 3, 2, 2, 2, 2, 0],
            [0, 4, 5, 2, 2, 6, 0],
            [0, 7, 5, 5, 8, 8, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    }

    fn get_noncontiguous_patch_grid() -> Array2<usize>{
        array![
            [0,  0,  0,  0,  0,  0,  0],
            [0,  1, 11, 11, 11, 11,  0],
            [0,  3, 11, 11, 11, 11,  0],
            [0,  4,  5, 11, 11,  6,  0],
            [0, 12,  5,  5,  8,  8,  0],
            [0,  0,  0,  0,  0,  0,  0],
        ]
    }

}