use std::rc::Rc;
use geo::{Geometry, Point};
//use rerun::RecordingStream;
use crate::model::layer::{CityLayer, CityLayerBuilder};
use crate::stream::{ChunkedRerunStream, Stream};

type RecordingStream = ChunkedRerunStream;

pub struct CityCenterLayer {
    city_center: Point,
}

impl CityCenterLayer {
    pub fn new(city_center: Point) -> Self {
        Self {
            city_center
        }
    }
}

impl CityLayer for CityCenterLayer {
    fn to_geojson(&self) -> String {
        todo!();
    }

    fn get_features(&self) -> Vec<Geometry> {
        vec![self.city_center.clone().into()]
    }
}

pub struct CityCenter {
    city_center: (usize, usize),
    stream: Option<Rc<RecordingStream>>
}


impl CityCenter {
    pub fn new(city_center: (usize, usize)) -> Self {
        Self {
            city_center,
            stream: None
        }
    }
}

impl CityLayerBuilder for CityCenter {
    fn with_rerun_stream(mut self: Box<Self>, stream: Rc<RecordingStream>) -> Box<dyn CityLayerBuilder> {
        self.stream = Some(stream);
        self
    }

    fn with_progress_bar(&self) {
        todo!()
    }

    fn build(&self, existing_layers: &Vec<Box<dyn CityLayer>>) -> Box<dyn CityLayer> {
        let city_center_pt = Point::new(self.city_center.0 as f64, self.city_center.1 as f64);
        self.stream.stream_geometry("city_center", &city_center_pt.clone().into());

        Box::new(CityCenterLayer::new(city_center_pt))
    }
}