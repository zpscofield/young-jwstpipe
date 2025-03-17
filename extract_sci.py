from astropy.io import fits
from astropy.wcs import WCS

img_data = fits.open('/Volumes/YOUNG6/BulletCluster/BulletCluster/stage3_output/F200W/output_files/BulletCluster_nircam_clear-F200W_i2d.fits')[1].data
img_header = fits.getheader('/Volumes/YOUNG6/BulletCluster/BulletCluster/stage3_output/F200W/output_files/BulletCluster_nircam_clear-F200W_i2d.fits', ext=1)
img_wcs = WCS(img_header)

hdu = fits.PrimaryHDU(img_data, header=img_header)
print(hdu)
hdu.writeto('/Volumes/YOUNG6/BulletCluster/BulletCluster/stage3_output/F200W/output_files/F200W_bullet_sci.fits', overwrite=True)
