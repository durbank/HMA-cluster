# Research article draft notes

The beginnings of the article I plan to write about the HMA clustering project.
This includes an outline and snippets of draft that I can transplant into a working document at a later date.

## Outline

### Introduction

Dicsuss the motivation and importance of HMA

- Water resources for large number of people
- Sea level rise?
- Understanding how changes in climate will affect different aspects of water resources

Background information and prior work

- Blah
- Blah
- Blah

Summarize my contributiona and how it fits in with/fill gaps in prior work

- Blah
- Blah
- Blah

### Methods

Data used

- Utilizes RGI and HAR v1 10-km climate reanalysis outputs (daily temperature, daily total precipitation, ?daily insolation?, all averaged 2000-2014)
- Calculate seasonal averages/totals of all variables, as well as annual averages/totals from HAR-based climatologies
- Extract seasonal and annual totals for the centroid of each glacier in RGI
- Handful (give number) of glaciers removed as outside the bounds of HAR and additional (give number) removed due to unrealistically low precipitation values (<1 mm total precipitation for climatological year)

HAR elevation and climate data corrections

- At times wide divergence of HAR elevation grid and measured median glacier elevation - we therefore correct model temperatures based on calculated lapse rates
- We cluster data based on climate variables, location, and elevation into 4 preliminary regions in order to determine cluster-specific temperature lapse rates (do the calculated clusters or lapse rates differ significantly if we exclude elevation data? elevation and location data?How can I quantitatively assess how much the clusters change? For lapse rates, I can merely calculate a percent change)
- We further calculate individual lapse rates for seasonal and annual temperatures (do these differ significantly?)
- We do not preform precipitation lapse rates due to the high variability in precipitation values within clusters, making it difficult to determine a lapse rate value with any degree of confidence (and due to the introduction on negative precipitation values in ~10% of results when linear precipitation lapse rates were introduced)
  - ?Could I instead try to calculate a precipitation scaling factor? In this way I could bound the result at 0 to avoid negative results...need to think about this more
- After correcting HAR temperatures due to elevation migration, we also introduce a new feature to the climate data - seasonal temperature amplitude. We calculate this as the difference between mean summer and mean winter temperature. As this is based on seasonal values, this metric is dampened compared to seasonal ampltude as calcuated on daily or finer temporal resolution data, but the relative differences in this metric within out data should still be consistent. It therefore is useable to perform clustering and classification work within our data

Climate region clustering

- We base climate clustering on seasonal and annual temperature and precipitation totals data
- We first normalize each variable to ensure equal weighting in our analysis
- We perform dimensionality reduction via principal component analysis, keeping the first **X** principal components, accounting for >95% of the total variance in our prescribed climate variables

Model comparisions

- Wish to try a variety of models and compare differences/similarities in the results between them
- I will want to use k-means, Gaussian mixture models?, hierchical clustering (agglomerative and/or divisive), and random forests