use geo::Point;
use crate::model::template::Template;
use crate::stream::RASTER_CELL_SIZE;

#[derive(Clone, Copy)]
pub struct Domain {
    minx: f64,
    miny: f64,
    maxx: f64,
    maxy: f64
}

impl Domain {
    pub fn from_template(template: &Template) -> Self {
        let minx = 0.0;
        let miny = 0.0;
        let maxx = template.clusters_street.shape()[0] as f64 * RASTER_CELL_SIZE as f64;
        let maxy = template.clusters_street.shape()[1] as f64 * RASTER_CELL_SIZE as f64;
        Domain{minx, miny, maxx, maxy}
    }

    pub fn is_inside(&self, pt: Point) -> bool {
        if pt.x() < self.minx || pt.x() > self.maxx || pt.y() < self.miny || pt.y() > self.maxy {
            return false;
        }
        return true
    }
}