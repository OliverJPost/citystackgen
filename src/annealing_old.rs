use rand::{Rng, thread_rng};
use ndarray::{Array, Array2, Ix2};
use rand::prelude::ThreadRng;
use gif::{Encoder, Frame, Repeat};
use image::{ImageBuffer, Rgb};
use std::fs::File;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use crate::{gridstats};
use crate::gridstats::{_get_patches, Aggregator, as_bins};

fn class_to_color(class: i32, palette: &HashMap<i32, Rgb<u8>>) -> Rgb<u8> {
    *palette.get(&class).unwrap_or(&Rgb([0, 0, 0])) // Default to black if the class isn't found
}

fn random_swap(mut array: &mut Array2<i32>, mut rng: &mut ThreadRng) {
    let rows = array.shape()[0];
    let row1 = rng.gen_range(0..rows);
    let row2 = rng.gen_range(0..rows);
    let cols = array.shape()[1];
    let col1 = rng.gen_range(0..cols);
    let col2 = rng.gen_range(0..cols);
    let index1 = (row1, col1);
    let index2 = (row2, col2);
    array.swap(index1, index2);
}


pub struct GridStatistics {
    patch_area: Vec<f64>,
    distances: Vec<f64>,
    core_area_index: Vec<f64>,
}

impl GridStatistics {
    pub fn new() -> GridStatistics {
        GridStatistics {
            patch_area: vec![],
            distances: vec![],
            core_area_index: vec![],
        }
    }
    pub fn from_indices(statistics: &GridStatistics, patch_indices: &Vec<i32>) -> GridStatistics {
        let patch_area = patch_indices.iter().map(|&patch_idx| statistics.patch_area[patch_idx as usize]).collect();
        let distances = patch_indices.iter().map(|&patch_idx| statistics.distances[patch_idx as usize]).collect();
        let core_area_index = patch_indices.iter().map(|&patch_idx| statistics.core_area_index[patch_idx as usize]).collect();
        GridStatistics {
            patch_area,
            distances,
            core_area_index,
        }
    }
}

enum Reverter {
    NotRevertable,
    SwapReverter((usize, usize), (usize, usize))
}

impl Reverter {
    fn revert(&mut self, grid: &mut GridLayer) {
        match self {
            Reverter::NotRevertable => panic!("Called revert on unrevertable grid"),
            Reverter::SwapReverter(index1, index2) => {
                grid.class_grid.swap(*index1, *index2);
                let affected_patches = grid.get_affected_patches(&[*index1, *index2]);
                grid.update_statistics(affected_patches);
            }
        }
    }
}

pub struct GridLayer {
    pub class_grid: Array2<i32>,
    patch_grid: Array2<i32>,
    statistics: GridStatistics,
    rows: usize,
    cols: usize,
    max_patch_idx: i32, // TODO usize
    available_patch_indices: VecDeque<i32>,
    unique_classes : Vec<i32>,
    reverter: Option<Reverter>,
}

pub(crate) fn build_patch_grid(domain: &Array2<i32>) -> (Array2<i32>, i32) {
    let (rows, cols) = domain.dim();
    let mut patch_grid = Array2::<i32>::from_elem((rows, cols), -1);
    let mut visited = Array2::<bool>::from_elem((rows, cols), false);

    let mut max_patch_idx = 0;
    for i in 0..rows {
        for j in 0..cols {
            let class = domain[[i, j]];
            if !visited[[i, j]] {
                crate::gridstats::flood_fill(domain, &mut patch_grid, &mut visited, i, j, class, max_patch_idx);
                max_patch_idx += 1;
            }
        }
    }
    (patch_grid, max_patch_idx - 1)
}

impl GridLayer {
    pub fn from_class_grid(class_grid: Array2<i32>) -> Self {
        let unique_classes = class_grid.iter().cloned().collect::<HashSet<i32>>().into_iter().collect();
        let (patch_grid, max_patch_idx) =  build_patch_grid(&class_grid);

        let rows = class_grid.shape()[0];
        let cols = class_grid.shape()[1];
        let mut grid = GridLayer{
            class_grid,
            patch_grid,
            statistics: GridStatistics::new(),
            rows,
            cols,
            max_patch_idx,
            available_patch_indices: VecDeque::new(),
            unique_classes,
            reverter: Some(Reverter::NotRevertable)
        };
        grid.update_statistics((0..=max_patch_idx).collect());
        grid
    }

    pub fn random_swap(&mut self, mut rng: &mut ThreadRng) {
        let index1 = self.get_random_index(&mut rng);
        let index2 = self.get_random_index(&mut rng);
        self.class_grid.swap(index1, index2);

        let affected_patches = self.get_affected_patches(&[index1, index2]);
        self.update_statistics(affected_patches);
        self.reverter = Some(Reverter::SwapReverter(index1, index2))
    }

    pub fn revert(&mut self) {
        match self.reverter.take() {
            Some(mut reverter) => reverter.revert(self),
            None => panic!("Called revert on grid without reverter")
        };
    }

    pub fn statistics_per_class(&self) -> HashMap<i32, GridStatistics> {
        let mut statistics_per_class = HashMap::new();

        let mut class_patch_indices: HashMap<i32, Vec<i32>> = HashMap::new();
        let mut visited_patches = HashSet::new();
        for row in 0..self.rows {
            for col in 0..self.cols {
                let patch_idx = self.patch_grid[(row, col)];
                if !visited_patches.contains(&patch_idx) {
                    visited_patches.insert(patch_idx);
                    let class_idx = self.class_grid[(row, col)];
                    match class_patch_indices.get_mut(&class_idx) {
                        Some(patch_indices) => patch_indices.push(patch_idx),
                        None => {
                            class_patch_indices.insert(class_idx, vec![patch_idx]);
                        }
                    }
                }
            }
        }

        for class in &self.unique_classes {
            let patch_indices = class_patch_indices.get(&class).unwrap();
            let statistics = GridStatistics::from_indices(&self.statistics, patch_indices);
            statistics_per_class.insert(*class, statistics);
        }
        statistics_per_class
    }

    fn get_affected_patches(&mut self, changed_cells: &[(usize, usize)]) -> HashSet<i32> {
        let mut affected_cells = vec![];
        let mut affected_patches = HashSet::new();
        let neighbours = [
            (-1, 0), // Up
            (1, 0), // Down
            (0, -1), // Left
            (0, 1), // Right
        ];
        for changed_cell in changed_cells {
            affected_cells.push(*changed_cell);
            affected_patches.insert(self.patch_grid[*changed_cell]);
            for &(di, dj) in &neighbours {
                let ni = (changed_cell.0 as i32 + di) as usize;
                let nj = (changed_cell.1 as i32 + dj) as usize;
                if ni < self.rows && nj < self.cols && ni >= 0 && nj >= 0{
                    affected_cells.push((ni, nj));
                    affected_patches.insert(self.patch_grid[(ni, nj)]);
                }
            }
        }
        let mut filled_patches = HashSet::new();
        let mut patch_id;
        let mut visited = Array2::<bool>::from_elem((self.rows, self.cols), false);
        for affected_cell in affected_cells {
            patch_id = self.patch_grid[affected_cell];
            // This patch idx has already been filled
            if filled_patches.contains(&patch_id) {
                if !visited[affected_cell] {
                    // But this cell hasn't been filled, so it must have become a new separate patch which needs a new id
                    patch_id = self.get_new_patch_id();
                    affected_patches.insert(patch_id);
                } else {
                    // And it has been filled, so no need to fill again
                    continue;
                }
            }

            self.flood_fill(affected_cell, &mut visited, patch_id);
            filled_patches.insert(self.patch_grid[affected_cell]);

        }
        let dropped_patches = affected_patches.difference(&filled_patches);
        self.available_patch_indices.extend(dropped_patches);
        affected_patches
    }

    fn get_new_patch_id(&mut self) -> i32 {
        let id = self.available_patch_indices.pop_front();
        match id {
            Some(patch_id) => patch_id,
            None => {
                self.max_patch_idx += 1;
                self.max_patch_idx
            }
        }
    }

    fn flood_fill(&mut self, start_cell: (usize, usize), visited: &mut Array2<bool>, patch_id: i32) {
        let mut stack = vec![start_cell];
        let class = self.class_grid[start_cell];

        while let Some((r, c)) = stack.pop() {
            if r < self.rows && c < self.cols && !visited[[r, c]] && self.class_grid[[r, c]] == class {
                visited[[r, c]] = true;
                self.patch_grid[[r, c]] = patch_id;
                // Explore the four possible directions
                if r > 0_usize { stack.push((r - 1, c)); }
                if r + 1_usize < self.rows { stack.push((r + 1, c)); }
                if c > 0_usize { stack.push((r, c - 1)); }
                if c + 1_usize < self.cols { stack.push((r, c + 1)); }
            }
        }
    }

    /// Due to dynamic updating, a lot of patch indices can become unused resulting in big
    /// gaps in the statistic vectors. This method makes all patch indices contiguous and
    /// updates the statistics
    fn condense_patch_indices(&mut self) {
        let mut patch_index_map = HashMap::new();
        let mut new_patch_idx = 0;
        for row in 0..self.rows {
            for col in 0..self.cols {
                let patch_idx = self.patch_grid[(row, col)];
                let new_patch_idx = match patch_index_map.get(&patch_idx) {
                    Some(new_idx) => *new_idx,
                    None => {
                        patch_index_map.insert(patch_idx, new_patch_idx);
                        new_patch_idx += 1;
                        new_patch_idx - 1
                    }
                };
                self.patch_grid[(row, col)] = new_patch_idx;
            }
        }
        self.available_patch_indices.clear();
        self.max_patch_idx = new_patch_idx;
        self.update_statistics((0..=new_patch_idx).collect());
    }

    fn update_statistics(&mut self, patches_to_update: HashSet<i32>) {
        let max_index = patches_to_update.iter().max().unwrap_or(&0) + 1; // Find the highest index that needs to be accessed

        // Ensure `patch_area` is large enough
        if self.statistics.patch_area.len() < max_index as usize {
            self.statistics.patch_area.resize(max_index as usize, 0.0); // Resize with default value, change 0 to a suitable default for `patch_area`
        }

        // Ensure `distances` is large enough
        if self.statistics.distances.len() < max_index as usize {
            self.statistics.distances.resize(max_index as usize, 0.0); // Resize with default value, change 0.0 to a suitable default for `distances`
        }

        if self.statistics.core_area_index.len() < max_index as usize {
            self.statistics.core_area_index.resize(max_index as usize, 0.0); // Resize with default value, change 0.0 to a suitable default for `core_area_index`
        }

        // Update statistics
        for patch_id in patches_to_update {
            self.statistics.patch_area[patch_id as usize] = patch_area(patch_id, &self.patch_grid);
            self.statistics.distances[patch_id as usize] = distance_to_center(patch_id, &self.patch_grid);
            self.statistics.core_area_index[patch_id as usize] = core_area_index(patch_id, &self.patch_grid);
        }
    }

    fn get_random_index(&mut self, rng: &mut &mut ThreadRng) -> (usize, usize) {
        let row = rng.gen_range(0..self.rows);
        let col = rng.gen_range(0..self.cols);
        let index1 = (row, col);
        index1
    }
}

fn core_area_index(patch_id: i32, patch_grid: &Array2<i32>) -> f64 {
    // TODO can be significantly faster
    let patch = patch_grid.mapv(|x| if x == patch_id { 1 } else { 0 });
    let (rows, cols) = patch_grid.dim();
    let mut total_area = 0;
    let mut core_area = 0;
    let neighbors = [
        (-1, 0), // Up
        (1, 0), // Down
        (0, -1), // Left
        (0, 1), // Right
    ];
    for i in 0..rows {
        for j in 0..cols {
            if patch[[i, j]] == 1 {
                total_area += 1;
                let mut is_core = true;
                for (di, dj) in &neighbors {
                    let ni = (i as i32 + di) as usize;
                    let nj = (j as i32 + dj) as usize;
                    if ni < rows && nj < cols && ni >= 0 && nj >= 0 {
                        if patch[[ni, nj]] == 0 {
                            is_core = false;
                            break;
                        }
                    }
                }
                if is_core {
                    core_area += 1;
                }
            }
        }
    }
    core_area as f64 / total_area as f64
}

fn distance_to_center(patch_id: i32, patch_grid: &Array2<i32>) -> f64 {
    let patch = patch_grid.mapv(|x| if x == patch_id { 1 } else { 0 });
    let (rows, cols) = patch_grid.dim();
    let center_row = rows / 2;
    let center_col = cols / 2;
    let mut distance = 0.0;
    let mut count = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            if patch[[i, j]] == 1 {
                distance += ((center_row as i32 - i as i32).abs() + (center_col as i32 - j as i32).abs()) as f64 * crate::gridstats::CELL_RESOLUTION;
                count += 1.0;
            }
        }
    }
    distance / count
}


fn patch_area(patch_id: i32, patch_grid: &Array2<i32>) -> f64 {
    let mut area = 0;
    for row in 0..patch_grid.shape()[0] {
        for col in 0..patch_grid.shape()[1] {
            if patch_grid[(row, col)] == patch_id {
                area += 1;
            }
        }
    }
    area as f64
}

// fn objective(array: &Array2<i32>) -> i64 {
//     let mut score= 0;
//     for window in array.windows((3,3)) {
//         let main_value = window[(1,1)];
//         for neighbour in [(0,1), (1, 0), (1, 2), (2, 1)] {
//             let value = window[neighbour];
//             if main_value == value {
//                 score += 5;
//             }
//         }
//     }
//     score
// }

enum Comparer {
    CosineSimilarity,
    Intersection
}

impl Comparer {
    fn compare(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        match self {
            Comparer::CosineSimilarity => cosine_similarity(a, b).unwrap(),
            Comparer::Intersection => intersection(a, b)
        }
    }
}

fn intersection(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let mut intersection = 0.0;
    let mut maximum_possible_intersection = 0.0;
    for (ai, bi) in a.iter().zip(b.iter()) {
        intersection += ai.min(*bi);
        maximum_possible_intersection += ai;
    }
    intersection / maximum_possible_intersection
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

fn similarity_by_distance(data1: &Vec<f64>, distances1: &Vec<f64>, data2: &Vec<f64>, distances2: &Vec<f64>, comparison_aggregator: Aggregator, comparer: &Comparer) -> f64 {
    let bins1 = as_bins(data1, distances1, 500.0, &comparison_aggregator);
    let bins2 = as_bins(data2, distances2, 500.0, &comparison_aggregator);
    comparer.compare(&bins1, &bins2)
}

fn objective(grid: &GridLayer, target: &HashMap<i32, GridStatistics>) -> f64 {
    let mut score= 0.;
    let metric_count = 4.;

    let comparer = Comparer::Intersection;

    for (class, class_statistics) in grid.statistics_per_class() {
        let target_statistics = target.get(&class).unwrap();
        score += similarity_by_distance(&class_statistics.patch_area, &class_statistics.distances, &target_statistics.patch_area, &target_statistics.distances, Aggregator::Average, &comparer);
        score += similarity_by_distance(&class_statistics.patch_area, &class_statistics.distances, &target_statistics.patch_area, &target_statistics.distances, Aggregator::Sum, &comparer);
        //score += similarity_by_distance(&class_statistics.patch_area, &class_statistics.distances, &target_statistics.patch_area, &target_statistics.distances, Aggregator::Max, &comparer);
        score += similarity_by_distance(&class_statistics.patch_area, &class_statistics.distances, &target_statistics.patch_area, &target_statistics.distances, Aggregator::Count, &comparer);
        //let bins1 = as_bins(&class_statistics.core_area_index, &class_statistics.distances, 500.0, &Aggregator::WeightedAverage(&class_statistics.distances));
        //let bins2 = as_bins(&target_statistics.core_area_index, &target_statistics.distances, 500.0, &Aggregator::WeightedAverage(&target_statistics.distances));
        //score += comparer.compare(&bins1, &bins2);
    }
    score * 1000. / metric_count
}

fn _objective(array: &Array2<i32>, target: &HashMap<i32, HashMap<&str, Vec<f64>>>) -> f64 {
    let mut score = 0.0;

    let patches = gridstats::_get_patches(array);
    for (class, class_patches) in patches {
        let class_targets = &target[&class];
        let distances = crate::gridstats::_distance_to_center(&class_patches);
        // let patch_core_area_index = crate::gridstats::_core_area_index(&class_patches);
        // let cai_bins = as_bins(&patch_core_area_index, &distances, 500.0, crate::gridstats::Aggregator::WeightedAverage(distances.to_owned()));
        // let target_cai_bins = &class_targets["cai_bins"];
        // score += cosine_similarity(&cai_bins, target_cai_bins).unwrap_or(0.0);
        let patch_areas = crate::gridstats::_area(&class_patches);
        let area_bins = as_bins(&patch_areas, &distances, 500.0, &crate::gridstats::Aggregator::Average);
        let target_area_bins = &class_targets["area_bins"];
        score += cosine_similarity(&area_bins, target_area_bins).unwrap_or(0.0);
    }

    // let rows = array.nrows();
    // let cols = array.ncols();
    // let center_row = rows / 2;
    // let center_col = cols / 2;
    //
    // for i in 1..rows - 1 {
    //     for j in 1..cols - 1 {
    //         let main_value = array[(i, j)];
    //         let neighbors = [
    //             (i - 1, j), // Up
    //             (i + 1, j), // Down
    //             (i, j - 1), // Left
    //             (i, j + 1), // Right
    //         ];
    //         for &(ni, nj) in &neighbors {
    //             if ni < rows && nj < cols && array[(ni, nj)] == main_value {
    //                 score += 5;
    //             }
    //         }
    //         // Reward classes 1 and 2 that are closer to the center
    //         if main_value == 1 || main_value == 2 {
    //             let distance = ((center_row as i32 - i as i32).abs() + (center_col as i32 - j as i32).abs()) as i64;
    //             let max_distance = (rows as i64 + cols as i64) / 2;
    //             score += (max_distance - distance) / 10;
    //         }
    //     }
    // }
    //
    // // Reward class 1 and 2 closer to the center


    score
}

fn print_progress(grid: &GridLayer, target: &HashMap<i32, GridStatistics>, comparer: &Comparer, verbose: bool) {
    let mut area_average_similarity = 0.0;
    let mut area_sum_similarity = 0.0;
    let mut area_count_similarity = 0.0;
    let mut core_area_index_similarity = 0.0;
    let mut i = 0;

    for (class, statistics) in grid.statistics_per_class() {
        let distances = &statistics.distances;
        let area = &statistics.patch_area;
        let target_distances = &target.get(&class).unwrap().distances;
        let target_area = &target.get(&class).unwrap().patch_area;
        let area_bins = as_bins(area, distances, 500., &Aggregator::Average);
        let target_area_bins = as_bins(target_area, target_distances, 500., &Aggregator::Average);
        area_average_similarity += comparer.compare(&area_bins, &target_area_bins);
        if verbose {
            println!("Area average");
            println!("{:?}", area_bins);
            println!("{:?}", target_area_bins);
        }
        let area_bins = as_bins(area, distances, 500., &Aggregator::Sum);
        let target_area_bins = as_bins(target_area, target_distances, 500., &Aggregator::Sum);
        area_sum_similarity += comparer.compare(&area_bins, &target_area_bins);
        if verbose {
            println!("Area sum");
            println!("{:?}", area_bins);
            println!("{:?}", target_area_bins);
        }
        let area_bins = as_bins(area, distances, 500., &Aggregator::Count);
        let target_area_bins = as_bins(target_area, target_distances, 500., &Aggregator::Count);
        area_count_similarity += comparer.compare(&area_bins, &target_area_bins);
        if verbose {
            println!("Area count");
            println!("{:?}", area_bins);
            println!("{:?}", target_area_bins);
        }
        i+=1;
        // let bins1 = as_bins(&statistics.core_area_index, &distances, 500.0, &Aggregator::WeightedAverage(&distances));
        // let bins2 = as_bins(&target.get(&class).unwrap().core_area_index, &target_distances, 500.0, &Aggregator::WeightedAverage(&target_distances));
        // core_area_index_similarity += comparer.compare(&bins1, &bins2);
    }
    println!("Area average: {:.2}. Sum: {:.2}. Count {:.2}. CAI {:.2}", area_average_similarity / i as f64, area_sum_similarity / i as f64, area_count_similarity / i as f64, core_area_index_similarity / i as f64);
}


pub fn _simulated_annealing(mut array: Array<i32, Ix2>, mut temperature: f64) -> Array<i32, Ix2>{
    let mut rng = thread_rng();

    let mut grid = GridLayer::from_class_grid(array);
    let mut target = grid.statistics_per_class();

    let final_temp = 0.001;
    let cooling_rate = 0.99999;

    let mut frames: Vec<Frame> = Vec::new();
    let gif_file_path = "output.gif";
    let gif_file = File::create(gif_file_path).expect("Unable to create GIF file.");
    let mut encoder = Encoder::new(gif_file, grid.cols as u16, grid.rows as u16, &[]).expect("Unable to create GIF encoder.");
    encoder.set_repeat(Repeat::Infinite).expect("Unable to set GIF to loop infinitely.");

    let comparer = Comparer::Intersection;

    let mut i = 0;
    // current time for perf counter
    let mut time = std::time::Instant::now();
    let mut current_score = objective(&grid, &target);
    let mut accepted = 0;
    let mut negative_accepted = 0;
    let mut rejected = 0;
    let mut overall_delta = 0.;
    while temperature > final_temp {
        grid.random_swap(&mut rng);
        let new_score = objective(&grid, &target);
        let delta = new_score - current_score;
        let probability = (delta as f64 / temperature).exp();

        if delta > 0.0 || rng.gen::<f64>() < probability {
            current_score = new_score;
            overall_delta += delta;
            accepted += 1;
            if delta < 0. {
                negative_accepted += 1;
            }
        } else {
            grid.revert();
            rejected += 1;
        }



        if i == 0 {
            println!("Start score: {}", current_score);
            print_progress(&grid, &target, &comparer, true);
        }

        temperature *= cooling_rate;
        if i % 5000 == 0 {
            println!("Temperature: {}. Score: {}. Delta: {}. Acc/rej {}/{}/{}", temperature, current_score, overall_delta, accepted, negative_accepted, rejected);
            print_progress(&grid, &target, &comparer, false);
            accepted = 0;
            negative_accepted = 0;
            rejected = 0;
            overall_delta = 0.;
            // iterations per second
            println!("Iterations per second: {:.0}", 5000. as f64 / time.elapsed().as_secs_f64());
            time = std::time::Instant::now();
        }
        let mut palette = HashMap::new();
        palette.insert(0, Rgb([255, 0, 0])); // Example class 0 mapped to Red
        palette.insert(1, Rgb([0, 255, 0])); // Example class 1 mapped to Green
        palette.insert(2, Rgb([0, 0, 255])); // Example class 2 mapped to Blue

        if i % 2000 == 0 {
            let (rows, cols) = (grid.rows, grid.cols);
            let mut img = ImageBuffer::new(cols as u32, rows as u32);

            // Safeguard array index access
            for (y, x, pixel) in img.enumerate_pixels_mut() {
                if (y as usize) < rows && (x as usize) < cols {
                    let class_value = grid.class_grid[(y as usize, x as usize)];
                    *pixel = class_to_color(class_value, &palette);
                } else {
                    *pixel = Rgb([0, 0, 0]); // Black (default for invalid indices)
                }
            }

            // Add frame to GIF
            let mut gif_frame = Frame::from_rgb(grid.cols as u16, grid.rows as u16, &img);
            gif_frame.delay = 10; // Adjust delay (in hundredths of a second)
            frames.push(gif_frame);
        }

        i += 1;
    }
    // Write all frames to the GIF file
    for frame in frames {
        encoder.write_frame(&frame).expect("Unable to write frame.");
    }

    println!("Final score: {}", current_score);
    print_progress(&grid, &target, &comparer, true);

    grid.class_grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_random() {
        let mut rng = thread_rng();
        let mut array = ndarray::Array2::zeros((25, 25));
        array.mapv_inplace(|_| rng.gen_range(0..2));

        _simulated_annealing(array, 0.00011);
    }
}