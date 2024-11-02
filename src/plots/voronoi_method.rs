use std::rc::Rc;
use geo::{BooleanOps, BoundingRect, EuclideanLength, GeometryCollection, LineInterpolatePoint, Point, Polygon};
use geos::Geom;
use geos::wkt::{ToWkt, Wkt};
use ndarray::Array2;
use voronoice::{BoundingBox, VoronoiBuilder};
use crate::model::grid::{TypologyGrid, TypologyGridX};
use crate::model::layer::{CityLayer, CityLayerBuilder};
use crate::model::template::Template;
use crate::plots::builder::PlotSystemBuilder;
use crate::plots::layer::PlotSystemLayer;
use crate::stream::{ChunkedRerunStream, Stream};
use crate::streets::builder::StreetNetworkBuilder;
use crate::streets::layer::StreetNetworkLayer;
use crate::streets::streets_experiment::{StreetNetwork, StreetNetworkBuilderX, vpt_to_co};

type RecordingStream = ChunkedRerunStream;

pub struct VoronoiPlotSystemBuilder {
    template: Template,
    rerun_stream: Option<Rc<RecordingStream>>
}

impl VoronoiPlotSystemBuilder {
    pub fn new(template: Template) -> Self {
        Self {
            template,
            rerun_stream: None
        }
    }
}

impl PlotSystemBuilder for VoronoiPlotSystemBuilder {
    fn generate_features(&self, typology_grid: &Box<dyn TypologyGrid>, existing_layers: &Vec<Box<dyn CityLayer>>) -> Vec<Polygon> {
        let street_segments = &existing_layers[1].get_features();
        let mut geometries: Vec<geos::Geometry> = vec![];
        for line in street_segments {
            let geom = geos::Geometry::new_from_wkt(&line.wkt_string()).unwrap();
            geometries.push(geom);
        }
        let re = geos::Geometry::polygonize(&geometries).unwrap();
        let wkt: Wkt<f64> = re.to_wkt().expect("Fail").parse().expect("Failed to parse WKT");

        let g: GeometryCollection = wkt.try_into().unwrap();
        let mut enclosures = vec![];
        for geom in g {
            let poly: Polygon = geom.try_into().unwrap();
            enclosures.push(poly)
        }


        let mut plots = vec![];
        for enclosure in &enclosures {
            let boundary = enclosure.exterior();
            self.rerun_stream.stream_geometry("vor/domain", &enclosure.clone().into());
            let bbox = boundary.bounding_rect().unwrap();
            let center = crate::streets::streets_experiment::co_to_vpt(bbox.center());
            let width = bbox.width();
            let height = bbox.height();
            let bbox2 = BoundingBox::new(center, width, height);
            let length = boundary.euclidean_length();
            let plot_width = 20.1;
            let steps = (length / plot_width).ceil() as usize;
            let mut points = vec![];
            for i in 0..steps {
                let point = boundary.line_interpolate_point(((i as f64) * plot_width) / length).unwrap();
                points.push(crate::streets::streets_experiment::pt_to_vpt(point));
            }
            for pt in &points {
                let gpt: Point = vpt_to_co(pt).into();
                self.rerun_stream.stream_geometry("vor/pts", &gpt.into())
            }

            if points.len() < 2 {
                plots.push(enclosure.clone());
                self.rerun_stream.stream_message("vor/log", "Too few points");
                continue;
            }

            let re = std::panic::catch_unwind(|| VoronoiBuilder::default()
                .set_sites(points)
                .set_bounding_box(bbox2)
                .build());

            if let Ok(Some(voronoi)) = re {
                let mut polys = vec![];
                let v = voronoi.vertices();
                for cell in voronoi.cells() {
                    let poly = geo::Polygon::new(cell.iter().map(|idx| crate::streets::streets_experiment::vpt_to_co(&v[*idx])).collect(), vec![]);
                    // intersection with enclosure
                    let r = std::panic::catch_unwind(|| poly.intersection(enclosure).0[0].clone());
                    match r {
                        Ok(poly) => {
                            polys.push(poly);
                        },
                        Err(_) => {
                            println!("Failed to create voronoi due to panic");
                            self.rerun_stream.stream_message("vor/log", "Failed to create voronoi. Panics!");
                        }
                    }
                }
                plots.extend(polys.clone());
                for poly in &polys {
                    self.rerun_stream.stream_geometry("vor/cells", &poly.clone().into())
                }
            } else {
                plots.push(enclosure.clone());
                self.rerun_stream.stream_message("vor/log", "Failed to create voronoi.");
                println!("Failed to create voronoi");
            }
        }
        plots
    }

}


impl CityLayerBuilder for VoronoiPlotSystemBuilder {


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