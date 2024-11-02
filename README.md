# CityStackGen
A Rust based CLI with Rerun.io GUI for generating cities based on templates encoded with [citypy](https://github.com/OliverJPost/citypy)

### Usage:
```bash
cargo run --release -- --template /Washington_dc_USA/Washington_dc_USA.npz --cluster-dir /Washington_dc_USA
```
With the template referencing the `.npz` file generated with `citypy` and cluster dir referencing the directory with the typology tempalte `.json` files generated by `citypy`.