use ndarray::Array2;
use std::collections::{HashMap, HashSet, VecDeque};
use rand::prelude::{IteratorRandom, ThreadRng};
use rand::Rng;
use crate::patchgridannealing::statistics::PatchGridStatistics;

pub struct PatchGrid {
    pub class_grid: Array2<i32>,
    pub(crate) patch_grid: Array2<usize>,
    pub statistics: PatchGridStatistics,
    rows: usize,
    cols: usize,
    max_patch_idx: usize,
    available_patch_indices: VecDeque<usize>,
    unique_classes : Vec<i32>,
    reverter: Option<Reverter>,
    affected_classes: HashSet<i32>
}

fn flood_fill(domain: &Array2<i32>, patch_map: &mut Array2<usize>, visited: &mut Array2<bool>, row: usize, col: usize, class: i32, patch_id: usize) {
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

fn construct_patch_grid(domain: &Array2<i32>) -> (Array2<usize>, usize) {
    let (rows, cols) = domain.dim();
    let mut patch_grid = Array2::<usize>::from_elem((rows, cols), usize::MAX);
    let mut visited = Array2::<bool>::from_elem((rows, cols), false);

    let mut max_patch_idx = 0_usize;
    for i in 0..rows {
        for j in 0..cols {
            let class = domain[[i, j]];
            if !visited[[i, j]] {
                flood_fill(domain, &mut patch_grid, &mut visited, i, j, class, max_patch_idx);
                max_patch_idx += 1;
            }
        }
    }

    (patch_grid, max_patch_idx - 1)
}

impl PatchGrid {
    pub fn init(class_grid: Array2<i32>, statistics: PatchGridStatistics) -> Self {
        let unique_classes = class_grid.iter().cloned().collect::<HashSet<i32>>().into_iter().collect();
        let (patch_grid, max_patch_idx) = construct_patch_grid(&class_grid);

        let rows = class_grid.shape()[0];
        let cols = class_grid.shape()[1];
        let mut grid = PatchGrid{
            class_grid,
            patch_grid,
            statistics,
            rows,
            cols,
            max_patch_idx,
            available_patch_indices: VecDeque::new(),
            unique_classes,
            reverter: Some(Reverter::NotRevertable),
            affected_classes: HashSet::new()
        };

        grid.statistics.update_patches(&(0..=max_patch_idx).collect(), &grid.patch_grid, &grid.class_grid);
        grid
    }

    pub fn random_swap(&mut self, mut rng: &mut ThreadRng) {
        let index1 = self.get_random_index(&mut rng);
        let index2 = self.get_random_index(&mut rng);
        self.class_grid.swap(index1, index2);

        let affected_patches= self.get_affected_patches(&[index1, index2]);
        self.statistics.update_patches(&affected_patches, &self.patch_grid, &self.class_grid);
        self.reverter = Some(Reverter::SwapReverter(index1, index2))
    }


    pub fn random_walk(&mut self, mut rng: &mut ThreadRng) {
        let index = self.get_random_index(&mut rng);
        let neighbor_offsets = [
            (-1, 0), // Up
            (1, 0), // Down
            (0, -1), // Left
            (0, 1), // Right
        ];
        let neighbours_indices = neighbor_offsets.iter().map(|(di, dj)| {
            let ni = (index.0 as i32 + di) as usize;
            let nj = (index.1 as i32 + dj) as usize;
            (ni, nj)
        }).collect::<Vec<_>>();
        let non_identical_neighbours = neighbours_indices.iter().filter(|(ni, nj)| {
            ni < &self.rows && nj < &self.cols && ni >= &0 && nj >= &0 && self.class_grid[(*ni, *nj)] != self.class_grid[index]
        }).collect::<Vec<_>>();
        if non_identical_neighbours.is_empty() {
            // Recursively pick a new random index if all neighbours are identical
            return self.random_walk(rng);
        }
        let index_to_swap = *non_identical_neighbours.into_iter().choose(&mut rng).unwrap();
        self.class_grid.swap(index, index_to_swap);

        let affected_patches = self.get_affected_patches(&[index, index_to_swap]);
        self.statistics.update_patches(&affected_patches, &self.patch_grid, &self.class_grid);
        self.reverter = Some(Reverter::SwapReverter(index, index_to_swap))
    }


    pub fn revert_state(&mut self) {
        match self.reverter.take() {
            Some(mut reverter) => reverter.revert(self),
            None => panic!("Called revert on grid without reverter")
        };
    }

    pub fn class_ids(&self) -> Vec<i32> {
        self.unique_classes.clone()
    }

    fn get_affected_patches(&mut self, changed_cells: &[(usize, usize)]) -> Vec<usize> {
        let mut affected_cells = vec![];
        let mut affected_patches = vec![];
        let neighbours = [
            (-1, 0), // Up
            (1, 0), // Down
            (0, -1), // Left
            (0, 1), // Right
        ];
        for changed_cell in changed_cells {
            affected_cells.push(*changed_cell);
            affected_patches.push(self.patch_grid[*changed_cell]);
            for &(di, dj) in &neighbours {
                let ni = (changed_cell.0 as i32 + di) as usize;
                let nj = (changed_cell.1 as i32 + dj) as usize;
                if ni < self.rows && nj < self.cols && ni >= 0 && nj >= 0{
                    affected_cells.push((ni, nj));
                    affected_patches.push(self.patch_grid[(ni, nj)]);
                }
            }
        }
        for affected_cell in &affected_cells {
            let class = self.class_grid[*affected_cell];
            self.affected_classes.insert(class);
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
                    affected_patches.push(patch_id);
                } else {
                    // And it has been filled, so no need to fill again
                    continue;
                }
            }

            self.flood_fill(affected_cell, &mut visited, patch_id);
            filled_patches.insert(self.patch_grid[affected_cell]);

        }
        affected_patches.sort();
        affected_patches.dedup();
        // let dropped_patches = affected_patches.difference(&filled_patches);
        let dropped_patches = affected_patches.iter().filter(|patch_id| !filled_patches.contains(patch_id)).cloned().collect::<Vec<usize>>();
        self.available_patch_indices.extend(dropped_patches);
        affected_patches
    }

    fn get_new_patch_id(&mut self) -> usize {
        let id = self.available_patch_indices.pop_front();
        match id {
            Some(patch_id) => patch_id,
            None => {
                self.max_patch_idx += 1;
                self.max_patch_idx
            }
        }
    }

    fn flood_fill(&mut self, start_cell: (usize, usize), visited: &mut Array2<bool>, patch_id: usize) {
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
        self.statistics.update_patches(&(0..=new_patch_idx).collect(), &self.patch_grid, &self.class_grid);
    }


    fn get_random_index(&mut self, rng: &mut &mut ThreadRng) -> (usize, usize) {
        let row = rng.gen_range(0..self.rows);
        let col = rng.gen_range(0..self.cols);
        let index1 = (row, col);
        index1
    }

    pub fn drain_affected_class_ids(&mut self) -> HashSet<i32> {
        let affected_classes = self.affected_classes.clone();
        self.affected_classes.clear();
        affected_classes
    }
}


enum Reverter {
    NotRevertable,
    SwapReverter((usize, usize), (usize, usize))
}

impl Reverter {
    fn revert(&mut self, grid: &mut PatchGrid) {
        match self {
            Reverter::NotRevertable => panic!("Called revert on unrevertable grid"),
            Reverter::SwapReverter(index1, index2) => {
                grid.class_grid.swap(*index1, *index2);
                let affected_patches = grid.get_affected_patches(&[*index1, *index2]);
                grid.statistics.update_patches(&affected_patches, &grid.patch_grid, &grid.class_grid);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn get_contiguous_class_grid() -> Array2<i32> {
        array![
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 2, 2, 2, 0],
            [0, 4, 2, 2, 2, 2, 0],
            [0, 1, 4, 2, 2, 1, 0],
            [0, 2, 4, 4, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    }

    fn get_noncontiguous_class_grid() -> Array2<i32> {
        array![
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 2, 2, 2, 0],
            [0, 4, 2, 2, 2, 2, 0],
            [0, 1, 4, 2, 2, 1, 0],
            [0, 2, 4, 4, 7, 7, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    }

    fn get_contiguous_patch_grid() -> Array2<usize> {
        array![
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 2, 2, 2, 0],
            [0, 3, 2, 2, 2, 2, 0],
            [0, 4, 5, 2, 2, 6, 0],
            [0, 7, 5, 5, 8, 8, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    }

    fn get_noncontiguous_patch_grid() -> Array2<usize> {
        array![
            [0,  0,  0,  0,  0,  0,  0],
            [0,  1, 11, 11, 11, 11,  0],
            [0,  3, 11, 11, 11, 11,  0],
            [0,  4,  5, 11, 11,  6,  0],
            [0, 12,  5,  5,  8,  8,  0],
            [0,  0,  0,  0,  0,  0,  0],
        ]
    }

    #[test]
    fn test_construct_patch_grid() {
        let grid = get_contiguous_class_grid();
        let (patch_grid, max_patch_idx) = construct_patch_grid(&grid);
        assert_eq!(max_patch_idx, 8);
        assert_eq!(patch_grid, get_contiguous_patch_grid());
        assert_ne!(patch_grid, get_noncontiguous_patch_grid());

        let grid = get_noncontiguous_class_grid();
        let (patch_grid, max_patch_idx) = construct_patch_grid(&grid);
        assert_eq!(max_patch_idx, 8);
        assert_eq!(patch_grid, get_contiguous_patch_grid());
    }

    #[test]
    fn test_random_swap() {
        let grid = get_contiguous_class_grid();
        let statistics = PatchGridStatistics::new();
        let mut patch_grid = PatchGrid::init(grid, statistics);
        let mut rng = rand::thread_rng();
        let class_grid = patch_grid.class_grid.clone();
        patch_grid.random_swap(&mut rng);
        assert_ne!(patch_grid.class_grid, class_grid);
    }

    #[test]
    fn test_revert() {
        let grid = get_contiguous_class_grid();
        let statistics = PatchGridStatistics::new();
        let mut patch_grid = PatchGrid::init(grid, statistics);
        let mut rng = rand::thread_rng();
        let class_grid = patch_grid.class_grid.clone();
        patch_grid.random_swap(&mut rng);
        patch_grid.revert_state();
        assert_eq!(patch_grid.class_grid, class_grid);
    }

    #[test]
    fn get_affected_patches_unmodified() {
        let grid = get_contiguous_class_grid();
        let statistics = PatchGridStatistics::new();
        let mut patch_grid = PatchGrid::init(grid, statistics);

        assert_eq!(patch_grid.patch_grid, get_contiguous_patch_grid());
        let mut affected_patches = patch_grid.get_affected_patches(&[(2, 4), (3,1)]);
        assert_eq!(patch_grid.patch_grid, get_contiguous_patch_grid());
        affected_patches.sort();
        assert_eq!(affected_patches, vec![0, 2, 3, 4, 5, 7]);

        // Swap class of a single cell patch, shouldn't modify anything in the patches
        patch_grid.class_grid[(1,1)] = 5;
        let mut affected_patches = patch_grid.get_affected_patches(&[(2, 4), (3,1)]);
        assert_eq!(patch_grid.patch_grid, get_contiguous_patch_grid());
        affected_patches.sort();
        assert_eq!(affected_patches, vec![0, 2, 3, 4, 5, 7]);
    }

    #[test]
    fn get_affected_patches_new_patch() {
        let grid = get_contiguous_class_grid();
        let statistics = PatchGridStatistics::new();
        let mut patch_grid = PatchGrid::init(grid, statistics);

        assert_eq!(patch_grid.patch_grid, get_contiguous_patch_grid());

        // New patch, no splits
        patch_grid.class_grid[(2,4)] = 1;
        let mut affected_patches = patch_grid.get_affected_patches(&[(2, 4)]);
        assert_eq!(patch_grid.patch_grid,
           array![
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 9, 9, 9, 9, 0],
                [0, 3, 9, 9, 2, 9, 0],
                [0, 4, 5, 9, 9, 6, 0],
                [0, 7, 5, 5, 8, 8, 0],
                [0, 0, 0, 0, 0, 0, 0],
        ]);
        affected_patches.sort();
        assert_eq!(affected_patches, vec![2, 9]);

        // Split patch in two


    }

    #[test]
    fn get_affected_patches_split_patch() {
        let grid = get_contiguous_class_grid();
        let statistics = PatchGridStatistics::new();
        let mut patch_grid = PatchGrid::init(grid, statistics);

        assert_eq!(patch_grid.patch_grid, get_contiguous_patch_grid());

        // Splits patch 5 in half
        patch_grid.class_grid[(4, 2)] = 1;
        let mut affected_patches = patch_grid.get_affected_patches(&[(4, 2)]);
        assert_eq!(patch_grid.patch_grid,
                   array![
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 2, 2, 2, 0],
                [0, 3, 2, 2, 2, 2, 0],
                [0, 4, 9, 2, 2, 6, 0],
                [0, 7, 5,10, 8, 8, 0],
                [0, 0, 0, 0, 0, 0, 0],
        ]);
        affected_patches.sort();
        assert_eq!(affected_patches, vec![0, 5, 7, 9, 10]);

        // Combine again
        patch_grid.class_grid[(4, 2)] = 4;
        let mut affected_patches = patch_grid.get_affected_patches(&[(4, 2)]);
        assert_eq!(patch_grid.patch_grid,
                   array![
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 2, 2, 2, 2, 0],
                [0, 3, 2, 2, 2, 2, 0],
                [0, 4, 5, 2, 2, 6, 0],
                [0, 7, 5, 5, 8, 8, 0],
                [0, 0, 0, 0, 0, 0, 0],
        ]);
        affected_patches.sort();
        assert_eq!(affected_patches, vec![0, 5, 7, 9, 10]);
    }
}