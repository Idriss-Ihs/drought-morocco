# Morocco Drought Monitoring

End-to-end pipeline to compute and map SPI/SPEI drought indices for Morocco (1981–present), using CHIRPS and ERA5-Land.

**Roadmap**

1. Ingest CHIRPS monthly precipitation → compute SPI (1/3/6/12).
2. Ingest ERA5-Land → compute PET → compute SPEI.
3. Aggregate to provinces; classify drought severity; build maps & time series.
