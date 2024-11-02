use std::rc::Rc;
//use rerun::RecordingStream;
use crate::model::domain::Domain;
use crate::model::layer::{CityLayer, CityLayerBuilder};
use crate::stream::ChunkedRerunStream;

type RecordingStream = ChunkedRerunStream;

pub struct CityBuilder{
    layer_builders: Vec<Box<dyn CityLayerBuilder>>,
    rerun_stream: Option<Rc<RecordingStream>>
}


impl CityBuilder {
    pub fn new() -> Self {
        CityBuilder {
            layer_builders: vec![],
            rerun_stream: None
        }
    }

    pub fn with_layer(mut self, layer_builder: Box<dyn CityLayerBuilder>) -> Self {
        // TODO Assert prerequisites
        self.layer_builders.push(layer_builder);
        self
    }

    pub fn with_rerun_stream(mut self, stream: Rc<ChunkedRerunStream>) -> Self {
        self.rerun_stream = Some(stream);
        self
    }

    pub fn build(self) -> City {
        let mut layers = vec![];
        for mut layer_builder in self.layer_builders {
            if let Some(stream) = &self.rerun_stream {
                layer_builder = layer_builder.with_rerun_stream(Rc::clone(&stream));
            }
            layers.push(layer_builder.build(&layers));
        }
        if let Some(stream) = &self.rerun_stream {
            stream.flush();
        }

        City {
            layers
        }
    }



}


pub struct City {
    layers: Vec<Box<dyn CityLayer>>
}