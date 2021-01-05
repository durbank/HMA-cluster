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
- Calculate seasonal averages/totals of all variables, as well as annual averages/totals from HAR-based climatologies (discuss/citations why HAR is used)
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

## Misc reference notes

Sakai and Fujita (2017)

- Used the techniques of Kaab et al (2015) to divide HMA into 10 sub-regions (I should see how these regions-and the techniques used to create them-compare to mine)
- Performed multiple regression analysis for mass balance sensitivity and found that 69% of the variance could be explained with summer temperature, monthly annual temperature amplitude, and summer precipitation as a fraction of total annual precipitation
- They found winter accumulation glaciers (summer precip <50% of total) have no strong mass balance sensitivity, but summer prcp-dominated glaciers have high variability in mass balance sensitivity
- They therefore contend that climatic settings represented by summer temp, annual temp range, and summer precip ratio are the dominant controls on the spatially contrasting mass change in Asian glaciers

Additional papers to read and assess

- [de Kok et. al., 2020](https://doi.org/10.5194/tc-14-3215-2020): Towards understanding the pattern of glacier mass balances in High Mountain Asia using regional climatic modelling
- [Zhao et. al., 2020](https://doi.org/10.1016/j.scitotenv.2020.140995): Spatiotemporal variability of glacier changes and their controlling factors in the Kanchenjunga region, Himalaya based on multi-source remote sensing data from 1975 to 2015
- [IPCC, 2019](http://e-space.mmu.ac.uk/623986/):  High Mountain Areas. In: IPCC Special Report on the Ocean and Cryosphere in a Changing Climate
- [Brun et. al., 2019]( https://doi.org/10.1029/2018JF004838): Heterogeneous Influence of Glacier Morphology on the Mass Balance Variability in High Mountain Asia
- [Azam et. al., 2018](https://doi.org/10.1017/jog.2017.86): Review of the status and mass changes of Himalayan-Karakoram glaciers
- [Kumar et. al., 2019](https://doi.org/10.1038/s41598-019-54553-9): Snowfall Variability Dictates Glacier Mass Balance Variability in Himalaya-Karakoram
- [Li et. al., 2019](https://doi.org/10.1016/j.accre.2020.03.003): Regional differences in global glacier retreat from 1980 to 2015
- [Farinotti et. al., 2020](https://doi.org/10.1038/s41561-019-0513-5): Manifestations and mechanisms of the Karakoram glacier Anomaly
- [Kumar et. al., 2020](https://doi.org/10.1016/j.quaint.2020.06.017): Glacier changes and associated climate drivers for the last three decades, Nanda Devi region, Central Himalaya, India
- [Wang et. al., 2019](https://doi.org/10.3390/w11040776): Spatial Heterogeneity in Glacier Mass-Balance Sensitivity across High Mountain Asia
- [Barandun et. al., 2020](https://doi.org/10.1016/j.wasec.2020.100072): The state and future of the cryosphere in Central Asia
- [Rao et. al., 2019](https://doi.org/10.1016/j.earscirev.2019.03.002): Reconciling the ‘westerlies’ and ‘monsoon’ models: A new hypothesis for the Holocene moisture evolution of the Xinjiang region, NW China
- [Johnson and Rupper, 2020](https://doi.org/10.3389/feart.2020.00129): An Examination of Physical Processes That Trigger the Albedo-Feedback on Glacier Surfaces and Implications for Regional Glacier Mass Balance Across High Mountain Asia
