Title: Physical properties of molecular clouds for the entire Milky Way disk 
Authors: Miville-Deschenes M.-A., Murray N., Lee E.J. 
Table: The molecular clouds catalog
================================================================================
Byte-by-byte Description of file: apjaa4dfdt1_mrt.txt
--------------------------------------------------------------------------------
   Bytes Format Units         Label  Explanations
--------------------------------------------------------------------------------
   1-  4 I4     ---           Cloud  Cloud number
   6-  8 I3     ---           Ncomp  Number of Gaussian components
  10- 12 I3     ---           Npix   Number of pixels on the sky
  14- 25 E12.6  deg2          A      Angular area
  27- 39 E13.6  deg           l      Baricentric Galactic longitude
  41- 52 E12.6  deg         e_l      Standard deviation in GLON
  54- 66 E13.6  deg           b      Baricentric Galactic latitude
  68- 79 E12.6  deg         e_b      Standard deviation in b
  81- 93 E13.6  deg           theta  ? Angle with respect to l = 0 (1)
  95-106 E12.6  K.km/s        WCO    Integrated CO emission
 108-119 E12.6  cm-2          NH2    Average column density
 121-132 E12.6  solMass/pc2   Sigma  Surface density
 134-146 E13.6  km/s          vcent  Centroid velocity
 148-159 E12.6  km/s          sigmav Velocity standard deviation
 161-172 E12.6  deg           Rmax   Largest eigenvalue of the inertia matrix
 174-185 E12.6  deg           Rmin   Smallest eigenvalue of the inertia matrix
 187-198 E12.6  deg           Rang   Angular size
 200-211 E12.6  kpc           Rgal   Galacto-centric radius
     213 I1     ---           INF    Near or far distance flag (2)
 215-226 E12.6  kpc           Dn     Near kinematic distance
 228-239 E12.6  kpc           Df     Far kinematic distance
 241-253 E13.6  kpc           zn     Near distance to Galactic mid-plane
 255-267 E13.6  kpc           zf     Far distance to Galactic mid-plane
 269-280 E12.6  pc2           Sn     Near derived physical area
 282-293 E12.6  pc2           Sf     Far derived physical area
 295-306 E12.6  pc            Rn     Near derived physical size
 308-319 E12.6  pc            Rf     Far derived physical size
 321-332 E12.6  solMass       Mn     Near derived mass
 334-345 E12.6  solMass       Mf     Far derived mass
--------------------------------------------------------------------------------
Note (1): Where 0.000000E+00 indicates a NULL value.
Note (2): Gives an estimate of which distance is more likely
          based on the {sigma}_v_ - {Sigma}R 
          relation ({sigma}_v_ = 0.23({Sigma}R)^0.43{+/-}0.14_; Eq. 26).
    0 = Near;
    1 = Far. 
--------------------------------------------------------------------------------
