import numpy as np
import geopandas as gpd


class BrazilRainforestBiomassEstimator:
    DA = 7.834  # as described in equation 9 of https://repositorio.inpa.gov.br/handle/1/37887
    DB = 3.467  # as described in equation 9 of https://repositorio.inpa.gov.br/handle/1/37887
    AGBA = 2.2737   # as described in equation 2 of https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0243079&type=printable
    AGBB = 1.9156   # as described in equation 2 of https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0243079&type=printable
    WC = 0.408  # water content, as described in equation 3 of https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0243079&type=printable
    CC = 0.485  # carbon content, as described in equation 3 of https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0243079&type=printable
    M13B1 = 0.183   # from table 3 of https://bg.copernicus.org/articles/13/1553/2016/bg-13-1553-2016-corrigendum.pdf
    M13B2 = 2.328   # from table 3 of https://bg.copernicus.org/articles/13/1553/2016/bg-13-1553-2016-corrigendum.pdf

    def estimate(self, polygon):
        diameter = (4 * polygon.area / np.pi) ** 0.5   # as described in equation 1 of https://repositorio.inpa.gov.br/handle/1/37887
        dbh = self.DA + self.DB * diameter             # as described in equation 9 of https://repositorio.inpa.gov.br/handle/1/37887

        # agb = self.AGBA * (dbh ** self.AGBB)           # as described in equation 2 of https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0243079&type=printable
        # agc = agb * (1 - self.WC) * self.CC            # as described in equation 3 of https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0243079&type=printable

        agb = self.M13B1 * (dbh ** self.M13B2)         # as described in table 2 of https://bg.copernicus.org/articles/13/1553/2016/bg-13-1553-2016.pdf
        agc = agb * (1 - self.WC) * self.CC            # as described in table 2 of https://bg.copernicus.org/articles/13/1553/2016/bg-13-1553-2016.pdf

        return diameter, dbh, agb, agc

    def estimate_gdf(self, gdf: gpd.GeoDataFrame):
        gdf['diameter'], gdf['dbh'], gdf['agb'], gdf['agc'] = zip(*gdf['geometry'].astype(object).apply(self.estimate))

        return gdf


