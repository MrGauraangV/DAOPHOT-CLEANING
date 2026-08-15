from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd


df = pd.read_csv("Summary of Stars.dat", delimiter=',')

with fits.open("RA Dec Reference Image.fits") as hdul:
    header = hdul[0].header
    wcs = WCS(header)


ra_deg, dec_deg = wcs.pixel_to_world_values(
    df["X_mean"].to_numpy(),
    df["Y_mean"].to_numpy()
)

coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)

df_out = pd.DataFrame({
    "star_id": df["Star"],
    "x_mean": df["X_mean"],
    "y_mean": df["Y_mean"],
    "ra": coords.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True),
    "dec": coords.dec.to_string(unit=u.deg, sep=":", precision=2, alwayssign=True, pad=True)
})

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
df_out.to_csv(
    "All_Data_RADEC_sexagesimal.dat",
    sep="\t",
    index=False
)