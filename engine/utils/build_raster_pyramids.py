from osgeo import gdal

if __name__ == '__main__':
    # Open the raster file
    input_rasters = [
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230606_cbblackburn5_p1/20230606_cbblackburn5_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230606_cbblackburn6_p1/20230606_cbblackburn6_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230607_cbblackburn2_p1/20230607_cbblackburn2_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230608_cbbernard1_p1/20230608_cbbernard1_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230608_cbbernard2_p1/20230608_cbbernard2_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230608_cbbernard3_p1/20230608_cbbernard3_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230608_cbbernard4_p1/20230608_cbbernard4_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230608_cbpapinas_p1/20230608_cbpapinas_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230712_afcagauthier_itrf20_p1/20230712_afcagauthier_itrf20_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230712_afcahoule_itrf20_p1/20230712_afcahoule_itrf20_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230712_afcamoisan_itrf20_p1/20230712_afcamoisan_itrf20_p1_rgb.cog.tif',
                     # 'D:/XPrize/Data/raw/quebec_plantations/20230817_afcagauthier_itrf20_p1/20230817_afcagauthier_itrf20_p1_rgb.cog.tif',
                        # 'D:/XPrize/Data/raw/quebec_plantations/20230817_afcahoule_itrf20_p1/20230817_afcahoule_itrf20_p1_rgb.cog.tif',
                     'C:/Users/Hugo/Documents/XPrize/data/raw/20240521_zf2100ha_highres_m3m_rgb_clahe_no_black_adjustedL_adjustedA1536_adjustedB1536.tif'
    ]

    for input_raster in input_rasters:
        dataset = gdal.Open(input_raster, gdal.GA_Update)

        if dataset is None:
            print("Failed to open the raster file.")
            exit(1)

        # Generate overviews
        resampling_method = gdal.GRA_Average  # Use 'average' resampling method
        overview_levels = [2, 4, 8, 16]        # Specify overview levels

        # Generate overviews at each specified level
        for level in overview_levels:
            dataset.BuildOverviews('NEAREST', [level])

        # Close the dataset
        dataset = None

