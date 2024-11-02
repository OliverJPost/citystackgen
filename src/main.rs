#![recursion_limit = "256"]

use std::rc::Rc;
use citypy_rust_utils_core::buildings::inset_method::BuildingSystemBuilderInset;
use citypy_rust_utils_core::model::domain::Domain;
use citypy_rust_utils_core::model::city::CityBuilder;
use citypy_rust_utils_core::model::template::Template;
use citypy_rust_utils_core::plots::oobb_method::PlotSystemBuilderOOBB;
use citypy_rust_utils_core::plots::voronoi_method::VoronoiPlotSystemBuilder;
use citypy_rust_utils_core::streets::builder::StreetNetworkBuilder;
use clap::Parser;
use citypy_rust_utils_core::buildings::combined_method::BuildingSystemBuilderCombined;
use citypy_rust_utils_core::buildings::irregular_method::BuildingSystemBuilderIrregular;
use citypy_rust_utils_core::buildings::oobb_method::BuildingSystemBuilderOOBB;
use citypy_rust_utils_core::buildings::place_method::BuildingSystemBuilderPlace;
use citypy_rust_utils_core::buildings::sweep_method::BuildingSystemBuilderSweep;
use citypy_rust_utils_core::enclosures::geos_method::EnclosureBuilderGEOS;
use citypy_rust_utils_core::context::city_center::CityCenter;
use citypy_rust_utils_core::stream::ChunkedRerunStream;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Filename of template file
    #[arg(short, long)]
    template: String,

    #[arg(short, long)]
    cluster_dir: String,

    #[arg(short, long)]
    no_annealing: bool,
}

fn main() {
    let args = Args::parse();

    let st = rerun::RecordingStreamBuilder::new("citypy")
        .spawn().unwrap();
    //let stream = Rc::new(st);
    let stream = Rc::new(ChunkedRerunStream::new(st, 1, true));
    let template = Template::from_file(&args.template, &args.cluster_dir).unwrap();
    let domain = Domain::from_template(&template);
    let city_center_row = template.city_center.0;
    let city_center_col = template.city_center.1;
    let tempalte_cell_size_m = 100;
    let city_center_xy = (city_center_col * tempalte_cell_size_m, city_center_row * tempalte_cell_size_m);
    let city_center = CityCenter::new(city_center_xy);
    let streets_builder = StreetNetworkBuilder::new(template.clone(), domain, args.no_annealing);
    // let plots_builder = PlotSystemBuilderOOBB::new(template.clone());
    let enclosure_builder = EnclosureBuilderGEOS::new(template.clone());
    let buildings_builder = BuildingSystemBuilderCombined::new(template,  args.no_annealing);
    let city = CityBuilder::new()
        .with_rerun_stream(stream)
        .with_layer(Box::new(city_center))
        .with_layer(Box::new(streets_builder))
        // .with_layer(Box::new(plots_builder))
        .with_layer(Box::new(enclosure_builder))
        .with_layer(Box::new(buildings_builder))
        .build();
}