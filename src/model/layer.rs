use std::rc::Rc;
use geo::Geometry;
use crate::model::domain::Domain;
use crate::model::template::Template;
use crate::stream::ChunkedRerunStream;

type RecordingStream = ChunkedRerunStream;

pub trait CityLayerBuilder {
    fn with_rerun_stream(self: Box<Self>, stream: Rc<RecordingStream>) -> Box<dyn CityLayerBuilder>;

    fn with_progress_bar(&self);


    fn build(&self, existing_layers: &Vec<Box<dyn CityLayer>>) -> Box<dyn CityLayer>;

}

pub trait CityLayer {
    fn to_geojson(&self) -> String;

    fn get_features(&self) -> Vec<Geometry>;
}