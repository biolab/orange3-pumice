- *station-data.pkl*

     Orange Table with data about stations:
     Code, Station, Country, Continent, Region, Latitude, Longitude, Elevation

     Country is obtained from station coordinates and may be wrong for stations
     on border (e.g. Turnu Severin is placed in Serbia, but is in Romania).

     Continent is smaller than a region; in particular, US is on continent
     North America and in region Americas.

- countries.pkl:
   Numpy array of shape |countries| x 2 with country names and continents.

   The array is not guaranteed to include all countries in station-data (but will never include additional countries), nor does it guarantee a particular order of countries.


Below, <prop> is tmin, tmax, tavg or prcp

The following three files contain numpy arrays with monthly means for stations,
computed over 2024, per decade or in total.
Rows are in the same order as in station-data.pkl.

- S-M2024-<prop>.pkl (~12 MB per file): |stations| x 12 months 
- S-MD-<prop>.pkl (~151 MB per file): |stations| x 12 months x 13 decades
- S-MT-<prop>.pkl (~13 MB per file): |stations| x 12 months

The following two files contain similar data, but averaged over all stations
in particular country. The order corresponds to countries.pkl.

- C-MD-<prop>.pkl (~0.272 M per file): |countries| x 12 months x 13 decades
- C-MT-<prop>.pkl (~0.021 M per file): |countries| x 12 months

The following file contains daily values for the last year (currently 2024)

- S-Y-<prop>.pkl (~350 MB per file): |stations| x 365 or 366 days