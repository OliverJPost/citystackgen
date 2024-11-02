use std::collections::VecDeque;
use std::panic::catch_unwind;
use std::rc::Rc;
use geo::{Area, BooleanOps, BoundingRect, Centroid, Coordinate, EuclideanDistance, EuclideanLength, GeometryCollection, Intersects, Line, LineInterpolatePoint, LineString, MinimumRotatedRect, Point, Polygon, Relate, Translate};
use geo::line_intersection::line_intersection;
use geo::LineIntersection::SinglePoint;
use geo_buffer::buffer_polygon;
use geos::Geom;
use geos::wkt::{ToWkt, Wkt};
use ndarray::Array2;
use ordered_float::Pow;
use petgraph::visit::Walker;
use rand::Rng;
// use rerun::{RecordingStream, TextLog};
use voronoice::{BoundingBox, VoronoiBuilder};
use crate::buildings::builder::BuildingSystemBuilder;
use crate::model::grid::{TypologyGrid, TypologyGridX};
use crate::model::layer::{CityLayer, CityLayerBuilder};
use crate::model::template::Template;
use crate::plots::builder::PlotSystemBuilder;
use crate::plots::layer::PlotSystemLayer;
use crate::stream::{ChunkedRerunStream, Stream};
use crate::streets::builder::StreetNetworkBuilder;
use crate::streets::layer::StreetNetworkLayer;
type RecordingStream = ChunkedRerunStream;

const AREA_THRESHOLD: f64 = 100.;

pub struct BuildingSystemBuilderIrregular {
    template: Template,
    rerun_stream: Option<Rc<RecordingStream>>
}

impl BuildingSystemBuilderIrregular {
    pub fn new(template: Template) -> Self {
        Self {
            template,
            rerun_stream: None
        }
    }
}

fn split_polygon_with_line(polygon: &Polygon<f64>, line: &Line<f64>) -> Vec<Polygon<f64>> {
    // Get the bounding box of the polygon
    let bbox = polygon.bounding_rect().unwrap();

    // Calculate the line's direction vector
    let dir_vector = Coordinate {
        x: line.end.x - line.start.x,
        y: line.end.y - line.start.y,
    };

    // Normalize the direction vector
    let length = (dir_vector.x.powi(2) + dir_vector.y.powi(2)).sqrt();
    let normalized_dir = Coordinate {
        x: dir_vector.x / length,
        y: dir_vector.y / length,
    };

    // Extend the line in both directions
    let extension = (bbox.width().powi(2) + bbox.height().powi(2)).sqrt() * 2.0;
    let extended_start = Coordinate {
        x: line.start.x - normalized_dir.x * extension,
        y: line.start.y - normalized_dir.y * extension,
    };
    let extended_end = Coordinate {
        x: line.end.x + normalized_dir.x * extension,
        y: line.end.y + normalized_dir.y * extension,
    };

    // Create two polygons from the extended line
    let perpendicular = Coordinate {
        x: -normalized_dir.y,
        y: normalized_dir.x,
    };

    let poly1 = Polygon::new(
        LineString::from(vec![
            extended_start,
            extended_end,
            Coordinate {
                x: extended_end.x + perpendicular.x * extension,
                y: extended_end.y + perpendicular.y * extension,
            },
            Coordinate {
                x: extended_start.x + perpendicular.x * extension,
                y: extended_start.y + perpendicular.y * extension,
            },
            extended_start,
        ]),
        vec![]
    );

    let poly2 = Polygon::new(
        LineString::from(vec![
            extended_start,
            extended_end,
            Coordinate {
                x: extended_end.x - perpendicular.x * extension,
                y: extended_end.y - perpendicular.y * extension,
            },
            Coordinate {
                x: extended_start.x - perpendicular.x * extension,
                y: extended_start.y - perpendicular.y * extension,
            },
            extended_start,
        ]),
        vec![]
    );

    // Perform boolean operations


    let result = catch_unwind(|| {
        let mut result: std::vec::Vec<Polygon> = vec![];
        result.extend(polygon.difference(&poly1).into_iter().filter_map(|mp| mp.try_into().ok()));


        result.extend(polygon.difference(&poly2).into_iter().filter_map(|mp| mp.try_into().ok()));
        result
    }
    );
    if result.is_err(){
        println!("Failed to split polygon");
        return vec![];
    }

    // Filter out any tiny polygons that might result from numerical imprecision
    result.unwrap().into_iter().filter(|p| p.unsigned_area() > 1e-10).collect()
}

impl BuildingSystemBuilder for BuildingSystemBuilderIrregular {
    fn generate_features(&self, typology_grid: &Box<dyn TypologyGrid>, existing_layers: &Vec<Box<dyn CityLayer>>) -> Vec<Polygon> {
        let enclosures = &existing_layers[2].get_features();
        let mut rng = rand::thread_rng();
        let mut buildings = vec![];
        let inset_enclosures: Vec<Polygon> = enclosures.iter().map(|enclosure| buffer_polygon(&Polygon::try_from(enclosure.clone()).unwrap(), -4.)).into_iter().flatten().collect();
        let double_enclosures: Vec<(Polygon, Polygon)> = inset_enclosures.into_iter().map(|item| (item.clone(), item)).collect();
        let mut polygons_to_split: VecDeque<(Polygon, Polygon)> = VecDeque::from(double_enclosures);

        const GAP_CHANCE: f64 = 0.2;
        while !polygons_to_split.is_empty() {
            let (poly, enclosure) = polygons_to_split.pop_front().unwrap();
            let oobb = poly.minimum_rotated_rect().unwrap();
            //self.rerun_stream.stream_geometry("buildings/poly", &poly.clone().into());
            //self.rerun_stream.stream_geometry("buildings/oobb", &oobb.clone().into());
            let long_edge_idx = oobb.exterior().lines().enumerate().max_by(|(_, a), (_, b)| a.euclidean_length().partial_cmp(&b.euclidean_length()).unwrap()).unwrap().0;
            let long_edge1 = oobb.exterior().lines().nth(long_edge_idx).unwrap();
            let long_edge2 = oobb.exterior().lines().nth((long_edge_idx + 2) % 4).unwrap();
            let fraction = rng.gen_range(0.3..0.7);
            let split_line_start = long_edge1.line_interpolate_point(fraction).unwrap();
            let split_line_end = long_edge2.line_interpolate_point(1.-fraction).unwrap();
            let split_line = Line::new(split_line_start, split_line_end);
            //self.rerun_stream.stream_geometry("buildings/split_line", &split_line.clone().into());
            let split_polygons = split_polygon_with_line(&poly, &split_line);


            for poly in split_polygons {
                if poly.unsigned_area() > AREA_THRESHOLD {
                    polygons_to_split.push_front((poly, enclosure.clone()));
                } else {
                    if rng.gen_bool(GAP_CHANCE) {
                        continue;
                    }
                    let max_dimension = poly.bounding_rect().unwrap().width().min(poly.bounding_rect().unwrap().height()) / 3.;
                    let random_inset;
                    if max_dimension > 1. {
                        random_inset = rng.gen_range(-max_dimension..-0.5);
                    } else {
                        random_inset = -0.5;
                    }
                    let inset_poly = buffer_polygon(&poly, random_inset);
                    for p in &inset_poly {
                        self.rerun_stream.stream_geometry("buildings/final", &p.clone().into());
                    }
                    buildings.extend(inset_poly)
                }
            }
        }
        buildings
    }

}


impl CityLayerBuilder for BuildingSystemBuilderIrregular {


    fn with_rerun_stream(mut self: Box<Self>, stream: Rc<RecordingStream>) -> Box<dyn CityLayerBuilder> {
        self.rerun_stream = Some(stream);
        self
    }

    fn with_progress_bar(&self) {
        todo!()
    }

    fn build(&self, existing_layers: &Vec<Box<dyn CityLayer>>) -> Box<dyn CityLayer> {
        let typology_grid = self.generate_typology_grid(&existing_layers);
        let plots = self.generate_features(&typology_grid, &existing_layers);
        Box::new(PlotSystemLayer::new(plots, typology_grid))
    }
}