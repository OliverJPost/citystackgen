use std::collections::HashMap;
use ndarray::{Array, Array2, Ix2};
use ndarray_stats::QuantileExt;
use plotly::{Histogram, Plot, Scatter};
use plotly::common::Mode;
use plotly::histogram::HistFunc;
use crate::gridstats;

pub(crate) const CELL_RESOLUTION: f64 = 100.0;

pub(crate) fn area(class_patch_map: &HashMap<i32, Array2<i32>>) -> HashMap<i32, Vec<f64>> {
    let mut areas = HashMap::new();
    for (class, class_patches) in class_patch_map {
        let patch_areas = _area(class_patches);

        areas.insert(class.to_owned(), patch_areas);
    }
    areas
}

fn _shrink_patches(patch_map: &Array2<i32>) -> Array2<i32> {
    let (rows, cols) = patch_map.dim();
    let mut reduced_patch_map = Array2::<i32>::from_elem((rows, cols), -1);
    let neighbors = [
        (-1, 0), // Up
        (1, 0), // Down
        (0, -1), // Left
        (0, 1), // Right
    ];
    // Add this cell to the reduced patch map if all the neighbouring cells belong to the same patch (so it's part of the core area)
    for i in 1..rows - 1 {
        for j in 1..cols - 1 {
            let patch_id = patch_map[[i, j]];
            let mut is_core = true;
            for &(di, dj) in &neighbors {
                let ni = (i as i32 + di) as usize;
                let nj = (j as i32 + dj) as usize;
                if patch_map[[ni, nj]] != patch_id {
                    is_core = false;
                    break;
                }
            }
            if is_core {
                reduced_patch_map[[i, j]] = patch_id;
            }
        }
    }
    reduced_patch_map
}

pub(crate) fn core_area_index(class_patch_map: &HashMap<i32, Array2<i32>>) -> HashMap<i32, Vec<f64>> {
    let mut areas = HashMap::new();
    for (class, class_patches) in class_patch_map {
        let patch_core_area_index = _core_area_index(class_patches);
        areas.insert(class.to_owned(), patch_core_area_index);
    }
    areas
}

pub(crate) fn _core_area_index(class_patches: &Array2<i32>) -> Vec<f64> {
    let patch_areas = _area(class_patches);
    let reduced_class_patches = _shrink_patches(class_patches);
    let patch_core_areas = _area(&reduced_class_patches);
    let patch_core_area_index = patch_areas.iter().zip(patch_core_areas.iter()).map(|(area, core_area)| {
        if *core_area == 0.0 {
            0.0  // or any other value you consider appropriate for cases where core_area is zero
        } else {
            *core_area / area
        }
    }).collect();
    patch_core_area_index
}

pub(crate) fn _area(class_patches: &Array2<i32>) -> Vec<f64> {
    let mut class_areas = vec![];
    let number_of_patches = class_patches.max().unwrap().to_owned();
    for path_idx in 0..=number_of_patches {
        let patch = class_patches.mapv(|x| if x == path_idx { 1 } else { 0 });
        let area = patch.sum() as f64 * CELL_RESOLUTION;
        class_areas.push(area);
    }
    class_areas
}


pub(crate) fn distance_to_center(class_patch_map: &HashMap<i32, Array2<i32>>) -> HashMap<i32, Vec<f64>> {
    let mut distances = HashMap::new();
    for (class, class_patches) in class_patch_map {
        let class_distances = _distance_to_center(class_patches);

        distances.insert(class.to_owned(), class_distances);
    }
    distances
}

pub(crate) fn _distance_to_center(class_patches: &Array2<i32>) -> Vec<f64> {
    let mut class_distances = vec![];
    let number_of_patches = class_patches.max().unwrap().to_owned();
    for path_idx in 0..=number_of_patches {
        let patch = class_patches.mapv(|x| if x == path_idx { 1 } else { 0 });
        let (rows, cols) = class_patches.dim();
        let center_row = rows / 2;
        let center_col = cols / 2;
        let mut distance = 0.0;
        let mut count = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                if patch[[i, j]] == 1 {
                    distance += ((center_row as i32 - i as i32).abs() + (center_col as i32 - j as i32).abs()) as f64 * CELL_RESOLUTION;
                    count += 1.0;
                }
            }
        }
        class_distances.push(distance / count); // FIXME divide by 0
    }
    class_distances
}


pub(crate) fn as_bins(data: &Vec<f64>, distances: &Vec<f64>, step: f64, aggregator: &Aggregator) -> Vec<f64> {
    let bins = aggregator.aggregate(data, distances, step);
    bins
}

pub(crate) fn create_target(grid: &Array2<i32>) -> HashMap<i32, HashMap<&str, Vec<f64>>> {
    let mut target = HashMap::new();
    let patches = gridstats::_get_patches(grid);
    for (class, class_patches) in patches {
        target.insert(class, HashMap::new());
        let distances = crate::gridstats::_distance_to_center(&class_patches);
        let patch_core_area_index = crate::gridstats::_core_area_index(&class_patches);
        //let cai_bins = as_bins(&patch_core_area_index, &distances, 500.0, &crate::gridstats::Aggregator::WeightedAverage(&distances));
        //target.get_mut(&class).unwrap().insert("cai_bins", cai_bins);
        let patch_areas = crate::gridstats::_area(&class_patches);
        let area_bins = as_bins(&patch_areas, &distances, 500.0, &crate::gridstats::Aggregator::Average);
        target.get_mut(&class).unwrap().insert("area_bins", area_bins);
    }
    target
}

pub(crate) enum Aggregator {
    Sum,
    Average,
   // WeightedAverage(&'a Vec<f64>),
    Max,
    Min,
    Count,
}

impl Aggregator {
    pub(crate) fn aggregate(&self, data: &Vec<f64>, distances: &Vec<f64>, step: f64) -> Vec<f64> {
        let max_distance = 50_000.0; // FIXME hardcoded
        let bin_count = (max_distance / step).ceil() as usize;
        let mut bins = vec![0.0; bin_count];
        match self {
            Aggregator::Sum => {
                for (distance, value) in distances.iter().zip(data) {
                    let bin = (distance / step).floor() as usize;
                    bins[bin] += value;
                }
                bins
            }
            Aggregator::Average => {
                let mut counts = vec![0.0; bin_count];
                for (distance, value) in distances.iter().zip(data) {
                    let bin = (distance / step).floor() as usize;
                    bins[bin] += value;
                    counts[bin] += 1.0;
                }
                bins.iter().zip(counts.iter()).map(|(bin, count)| if *count != 0.0 { bin / count } else { 0.0 }).collect()
            }
            // Aggregator::WeightedAverage(weights) => {
            //     let mut counts = vec![0.0; bin_count];
            //
            //     for ((distance, value), weight) in distances.iter().zip(data).zip(*weights) {
            //         let bin = (distance / step).floor() as usize;
            //         bins[bin] += value * weight;
            //         counts[bin] += weight;
            //     }
            //     bins.iter().zip(counts.iter()).map(|(bin, count)| if *count != 0.0 { bin / count } else { 0.0 }).collect()
            // }
            Aggregator::Max => {
                let mut max_value = 0.0;
                for (distance, value) in distances.iter().zip(data) {
                    let bin = (distance / step).floor() as usize;
                    if *value > max_value {
                        bins[bin] = *value;
                        max_value = *value;
                    }
                }
                bins
            }
            Aggregator::Min => {
                panic!()
            }
            Aggregator::Count => {
                for distance in distances {
                    let bin = (distance / step).floor() as usize;
                    bins[bin] += 1.0;
                }
                bins
            }
        }
    }
}

pub fn _get_patches(domain: &Array2<i32>) -> HashMap<i32, Array2<i32>> {
    let mut patches = HashMap::new();
    let (rows, cols) = domain.dim();
    let mut visited = Array2::<bool>::from_elem((rows, cols), false);

    for i in 0..rows {
        for j in 0..cols {
            let class = domain[[i, j]];
            if !visited[[i, j]] {
                // Each class gets its own matrix, initialized to -1
                let patch_map = patches.entry(class).or_insert_with(|| Array2::<i32>::from_elem((rows, cols), -1));
                // Start a new patch ID
                let patch_id = (patch_map.iter().filter(|&&x| x != -1).max().unwrap_or(&-1)) + 1;
                flood_fill(domain, patch_map, &mut visited, i, j, class, patch_id);
            }
        }
    }

    // let areas = area(&patches);
    // println!("{:?}", areas);
    // let distances = distance_to_center(&patches);
    // println!("{:?}", distances);
    // let cai = core_area_index(&patches);
    // println!("{:?}", cai);
    //
    // // plot distance to area
    // let mut plot;
    //
    // for (class, class_distances) in distances {
    //     plot = Plot::new();
    //
    //     let class_areas = &areas[&class];
    //     let bins = as_bins(class_areas, &class_distances, 500.0, Aggregator::Sum);
    //     println!("AREA {:?}", class);
    //     println!("{:?}", bins);
    //
    //     let patch_cai = &cai[&class];
    //     let bins = as_bins(patch_cai, &class_distances, 500.0, Aggregator::WeightedAverage(class_areas.to_owned()));
    //     println!("CAI {:?}", class);
    //     println!("{:?}", bins);
    //
    //     let trace = Histogram::new_xy(class_distances, class_areas.to_owned()).name("h").hist_func(HistFunc::Average);
    //     //let trace = Scatter::new(class_distances, class_areas.to_owned()).mode(Mode::Markers);
    //     plot.add_trace(trace);
    //
    //     plot.write_html(format!("class_{}.html", class));
    // }

    patches
}

pub(crate) fn flood_fill(domain: &Array2<i32>, patch_map: &mut Array2<i32>, visited: &mut Array2<bool>, row: usize, col: usize, class: i32, patch_id: i32) {
    let (rows, cols) = domain.dim();
    let mut stack = vec![(row, col)];

    while let Some((r, c)) = stack.pop() {
        if r < rows && c < cols && !visited[[r, c]] && domain[[r, c]] == class {
            visited[[r, c]] = true;
            patch_map[[r, c]] = patch_id;
            // Explore the four possible directions
            if r > 0 { stack.push((r - 1, c)); }
            if r + 1 < rows { stack.push((r + 1, c)); }
            if c > 0 { stack.push((r, c - 1)); }
            if c + 1 < cols { stack.push((r, c + 1)); }
        }
    }
}