# CityStackGen
A Rust based CLI with Rerun.io GUI for generating cities based on templates encoded with [citypy](https://github.com/OliverJPost/citypy)

Part of the TU Delft MSc Geomatics Thesis "The City Stack - A Morphology-Based City Analysis and Generation Framework". [Repo](https://github.com/OliverJPost/CityStack)

### Usage:
```bash
cargo run --release -- --template /Washington_dc_USA/Washington_dc_USA.npz --cluster-dir /Washington_dc_USA
```
With the template referencing the `.npz` file generated with `citypy` and cluster dir referencing the directory with the typology tempalte `.json` files generated by `citypy`.