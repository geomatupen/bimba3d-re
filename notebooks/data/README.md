This folder stores prepared training-data artifacts used by the standalone notebooks.

Each dataset folder should contain:

- `manifest.json`: dataset summary, descriptor schema, multiplier bounds, base parameter values, and build metadata.
- `rows.json`: prepared training rows used by the Ridge and MLP notebooks.

The notebooks are configured to read from this folder so they can run without depending on the application data directory.
