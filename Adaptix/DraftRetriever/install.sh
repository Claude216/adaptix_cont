RUSTFLAGS="-C target-cpu=native" maturin build --release --strip -i python3.10
pip install target/wheels/draftretriever-0.1.0-cp310-cp310-linux_x86_64.whl --force-reinstall
