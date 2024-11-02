use ndarray::Array2;

pub trait TypologyGrid{
    fn get_class_grid(&self) -> &Array2<i32>;

    fn interpolate(&self, x: f64, y: f64) -> i32 {
        todo!()
    }
}

pub struct TypologyGridX {
    pub(crate) class_grid: Array2<i32>,

}

impl TypologyGrid for TypologyGridX {
    fn get_class_grid(&self) -> &Array2<i32> {
        &self.class_grid
    }
}