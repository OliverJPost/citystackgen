use ndarray::Array2;

pub trait ClassMetric{

}

pub trait PatchMetric: Send + Sync  {
    fn patch_data(&self) -> &Vec<f64>;
    fn patch_data_mut(&mut self) -> &mut Vec<f64>;
    fn _update(&mut self, patch_ids: &Vec<usize>, patch_grid: &Array2<usize>, patch_ranges: &Vec<Option<((usize, usize),(usize,usize))>>);
    fn id(&self) -> PatchGridMetricID;

    fn update(&mut self, patch_ids: &Vec<usize>, patch_grid: &Array2<usize>, patch_ranges: &Vec<Option<((usize, usize),(usize,usize))>>) {
        let max_patch_id = patch_ids.iter().max().unwrap();
        if self.patch_data_mut().len() <= *max_patch_id {
            self.patch_data_mut().resize(max_patch_id + 1, 0.0);
        }
        self._update(patch_ids, patch_grid, patch_ranges)
    }

    fn values(&self, patch_ids: &Vec<usize>) -> Vec<f64> {
        patch_ids.iter().map(|patch_id| self.patch_data()[*patch_id]).collect()
    }
}

/// From https://fragstats.org/index.php/fragstats-metrics/patch-based-metrics/shape-metrics/p2-shape-index
pub struct PatchShapeIndex {
    patch_data: Vec<f64>
}

impl PatchShapeIndex {
    pub fn new() -> Self {
        PatchShapeIndex {
            patch_data: vec![]
        }
    }

    fn shape_index(patch_id: usize, patch_grid: &Array2<usize>, patch_range: ((usize, usize), (usize, usize))) -> f64 {
        let perimeter = patch_perimeter(patch_id, patch_grid, patch_range);
        let area = patch_area(patch_id, patch_grid, patch_range);
        (0.25 * perimeter)/area.sqrt()
    }
}

fn patch_perimeter(patch_id: usize, patch_grid: &Array2<usize>, patch_range: ((usize, usize), (usize, usize))) -> f64 {
    let mut perimeter = 0;
    let neighbor_offsets = vec![(0, 1), (1, 0), (0, -1), (-1, 0)];
    let ((min_row, max_row), (min_col, max_col)) = patch_range;
    for row in min_row..=max_row {
        for col in min_col..=max_col {
            if patch_grid[(row, col)] == patch_id {
                for (row_offset, col_offset) in &neighbor_offsets {
                    let neighbor_row = row as i32 + row_offset;
                    let neighbor_col = col as i32 + col_offset;
                    if neighbor_row < 0 || neighbor_col < 0 || neighbor_row >= patch_grid.shape()[0] as i32 || neighbor_col >= patch_grid.shape()[1] as i32 {
                        perimeter += 1;
                    } else if patch_grid[(neighbor_row as usize, neighbor_col as usize)] != patch_id {
                        perimeter += 1;
                    }
                }
            }
        }
    }
    perimeter as f64
}

impl PatchMetric for PatchShapeIndex {
    fn patch_data(&self) -> &Vec<f64> {
        &self.patch_data
    }

    fn patch_data_mut(&mut self) -> &mut Vec<f64> {
        &mut self.patch_data
    }

    fn _update(&mut self, patch_ids: &Vec<usize>, patch_grid: &Array2<usize>, patch_ranges: &Vec<Option<((usize, usize),(usize,usize))>>) {
        for patch_id in patch_ids {
            let patch_range = patch_ranges[*patch_id].unwrap();
            self.patch_data[*patch_id] = PatchShapeIndex::shape_index(*patch_id, patch_grid, patch_range);
        }
    }

    fn id(&self) -> PatchGridMetricID {
        PatchGridMetricID::PatchShapeIndex
    }
}

pub struct PatchRelatedCircumscribingCircle {

}

pub struct PatchContiguityIndex {

}

pub struct PatchArea {
    patch_data: Vec<f64>
}

fn patch_area(patch_id: usize, patch_grid: &Array2<usize>, patch_range: ((usize, usize), (usize, usize))) -> f64 {
    let mut area = 0;
    let ((min_row, max_row), (min_col, max_col)) = patch_range;
    for row in min_row..=max_row {
        for col in min_col..=max_col {
            if patch_grid[(row, col)] == patch_id {
                area += 1;
            }
        }
    }
    area as f64
}

impl PatchArea {
    pub fn new() -> Self {
        PatchArea {
            patch_data: vec![]
        }
    }
}

impl PatchMetric for PatchArea {
    fn patch_data(&self) -> &Vec<f64> {
        &self.patch_data
    }
    fn patch_data_mut(&mut self) -> &mut Vec<f64> {
        &mut self.patch_data
    }

    fn _update(&mut self, patch_ids: &Vec<usize>, patch_grid: &Array2<usize>, patch_ranges: &Vec<Option<((usize, usize),(usize,usize))>>) {
        for patch_id in patch_ids {
            let patch_range = patch_ranges[*patch_id].unwrap();
            self.patch_data[*patch_id] = patch_area(*patch_id, patch_grid, patch_range);
        }
    }


    fn id(&self) -> PatchGridMetricID {
        PatchGridMetricID::PatchArea
    }
}

pub struct PatchRadiusOfGyration{
    patch_data: Vec<f64>
}


pub struct PatchCoreAreaIndex {
    patch_data: Vec<f64>
}

impl PatchCoreAreaIndex {
    pub fn new() -> Self {
        PatchCoreAreaIndex {
            patch_data: vec![]
        }
    }
}

fn patch_core_area(patch_id: usize, patch_grid: &Array2<usize>, patch_range: ((usize, usize), (usize, usize))) -> f64 {
    let mut core_area = 0;
    let neighbor_offsets = vec![(0, 1), (1, 0), (0, -1), (-1, 0)];
    let ((min_row, max_row), (min_col, max_col)) = patch_range;
    for row in min_row..=max_row {
        for col in min_col..=max_col {
            if patch_grid[(row, col)] == patch_id {
                let mut is_core = true;
                for (row_offset, col_offset) in &neighbor_offsets {
                    let neighbor_row = row as i32 + row_offset;
                    let neighbor_col = col as i32 + col_offset;
                    if neighbor_row < 0 || neighbor_col < 0 || neighbor_row >= patch_grid.shape()[0] as i32 || neighbor_col >= patch_grid.shape()[1] as i32 {
                        continue;
                    }
                    if patch_grid[(neighbor_row as usize, neighbor_col as usize)] != patch_id {
                        is_core = false;
                        break;
                    }
                }
                if is_core {
                    core_area += 1;
                }
            }
        }
    }
    core_area as f64
}


impl PatchMetric for PatchCoreAreaIndex {
    fn patch_data(&self) -> &Vec<f64> {
        &self.patch_data
    }
    fn patch_data_mut(&mut self) -> &mut Vec<f64> {
        &mut self.patch_data
    }

    fn _update(&mut self, patch_ids: &Vec<usize>, patch_grid: &Array2<usize>, patch_ranges: &Vec<Option<((usize, usize),(usize,usize))>>) {
        for patch_id in patch_ids {
            let patch_range = patch_ranges[*patch_id].unwrap();
            let patch_area = patch_area(*patch_id, patch_grid, patch_range);
            let core_area = patch_core_area(*patch_id, patch_grid, patch_range);
            self.patch_data[*patch_id] = core_area / patch_area;
        }
    }

    fn id(&self) -> PatchGridMetricID {
        PatchGridMetricID::PatchCoreAreaIndex
    }
}

pub struct PatchDistanceToCenter {
    patch_data: Vec<f64>,
    center_point: (f64, f64),
    cell_size: f64
}

fn distance_to_cell(patch_id: usize, patch_grid: &Array2<usize>, center_cell: (usize, usize), patch_range: ((usize, usize), (usize, usize))) -> f64 {
    let center_row = center_cell.0;
    let center_col = center_cell.1;
    let mut distance = 0.0;
    let mut count = 0.0;
    let ((min_row, max_row), (min_col, max_col)) = patch_range;
    for i in min_row..=max_row {
        for j in min_col..=max_col {
            if patch_grid[[i, j]] == patch_id {
                distance += (((center_row as i32 - i as i32).pow(2) + (center_col as i32 - j as i32).pow(2)) as f64).sqrt();
                count += 1.0;
            }
        }
    }
    distance / count
}
impl PatchDistanceToCenter {
    pub fn new(center_point: (f64, f64), cell_size: f64) -> Self {
        PatchDistanceToCenter {
            patch_data: vec![],
            center_point,
            cell_size
        }
    }
}

impl PatchMetric for PatchDistanceToCenter {
    fn patch_data(&self) -> &Vec<f64> {
        &self.patch_data
    }
    fn patch_data_mut(&mut self) -> &mut Vec<f64> {
        &mut self.patch_data
    }

    fn _update(&mut self, patch_ids: &Vec<usize>, patch_grid: &Array2<usize>, patch_ranges: &Vec<Option<((usize, usize),(usize,usize))>>) {
        for patch_id in patch_ids {
            let patch_range = patch_ranges[*patch_id].unwrap();
            let center_cell = ((self.center_point.1 / self.cell_size) as usize, (self.center_point.0 / self.cell_size) as usize);
            self.patch_data[*patch_id] = distance_to_cell(*patch_id, patch_grid, center_cell, patch_range) * self.cell_size;
        }
    }

    fn id(&self) -> PatchGridMetricID {
        PatchGridMetricID::PatchDistanceToCenter
    }
}


#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub enum PatchGridMetricID {
    PatchArea,
    PatchDistanceToCenter,
    PatchCoreAreaIndex,
    PatchShapeIndex
}

#[cfg(test)]
mod tests {
    use approx::{assert_relative_eq, relative_eq};
    use super::*;
    use ndarray::{array, Array2};

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
    #[test]
    fn test_patch_basics() {
        let mut patch_grid = get_contiguous_patch_grid();
        // Range of 0,0 to 5,6 for all
        let fake_ranges = vec![Some(((0, 5), (0, 6))); 13];

        let mut metric = PatchArea::new();
        metric.update(&(0..=8).collect(), &patch_grid, &fake_ranges);
        assert_eq!(metric.patch_data, vec![22., 1., 10., 1., 1., 3., 1., 1., 2.]);

        // Swap a patch
        patch_grid[(1, 2)] = 1;
        metric.update(&(0..=8).collect(), &patch_grid, &fake_ranges);
        assert_eq!(metric.patch_data, vec![22., 2., 9., 1., 1., 3., 1., 1., 2.]);

        // Drop a patch
        patch_grid[(3, 5)] = 2;
        metric.update(&(0..=8).collect(), &patch_grid, &fake_ranges);
        assert_eq!(metric.patch_data, vec![22., 2., 10., 1., 1., 3., 0., 1., 2.]);

        // Add a patch
        patch_grid[(3, 5)] = 9;
        metric.update(&(0..=9).collect(), &patch_grid, &fake_ranges);
        assert_eq!(metric.patch_data, vec![22., 2., 9., 1., 1., 3., 0., 1., 2., 1.]);

        // Add a noncontiguous patch id
        patch_grid[(3, 5)] = 12;
        metric.update(&(0..=12).collect(), &patch_grid, &fake_ranges);
        assert_eq!(metric.patch_data, vec![22., 2., 9., 1., 1., 3., 0., 1., 2., 0., 0., 0., 1.]);
    }

    #[test]
    fn test_patch_distance_to_center() {
        let mut patch_grid = get_contiguous_patch_grid();
        let mut metric = PatchDistanceToCenter::new((2.0, 3.0), 1.);
        let fake_ranges = vec![Some(((0, 5), (0, 6))); 9];

        metric.update(&(0..=8).collect(), &patch_grid, &fake_ranges);
        // Single cell patches
        assert_eq!(metric.patch_data[4], 1.); // 1 left
        assert_relative_eq!(metric.patch_data[7], 1.414, epsilon = 0.01); // 1 left 1 down

        // Multicell patches
        assert_relative_eq!(metric.patch_data[5], (1.414 + 1.)/3., epsilon = 0.01); // 1 left 1 down + 1 down + in center
    }

    #[test]
    fn test_patch_corea_area_index() {
        let mut patch_grid = get_contiguous_patch_grid();
        let mut metric = PatchCoreAreaIndex::new();
        let fake_ranges = vec![Some(((0, 5), (0, 6))); 9];

        metric.update(&(0..=8).collect(), &patch_grid, &fake_ranges);
        assert_relative_eq!(metric.patch_data[5], 0., epsilon = 0.01); // No interior cells
        assert_relative_eq!(metric.patch_data[4], 0., epsilon = 0.01); // Single cell
        assert_relative_eq!(metric.patch_data[2], 2./10., epsilon = 0.01); // Area 10, Core 2

    }

    #[test]
    fn test_get_values() {
        let mut patch_grid = get_contiguous_patch_grid();
        let fake_ranges = vec![Some(((0, 5), (0, 6))); 9];
        let mut metric = PatchArea::new();
        metric.update(&(0..=8).collect(), &patch_grid, &fake_ranges);
        assert_eq!(metric.values(&vec![0, 2, 8]), vec![22., 10., 2.]);

    }
}