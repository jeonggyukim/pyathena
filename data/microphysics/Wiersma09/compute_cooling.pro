function compute_cooling, tablepath, tabletype, redshift, density, temperature, AH, AHe, AC, AN, AO, ANe, AMg, ASi, AS, ACa, AFe, massfrac, help=help

; This routine computes net cooling for a given set of abundances. See
; test = compute_cooling(/help)
; for more details

; This is version 100423.
; Change history:
;      * 08????: Original version
;      * 090512: Fixed issue regarding checking the boundaries of the
;                redshift array, updated reference.
;      * 100423: Clarified help
;
; Please refer to Wiersma, Schaye, & Smith 2009, MNRAS, 393, 99
; for a description of the methods used.
; 
; For bug reports and questions, please contact 
; Rob Wiersma <wiersma@strw.leidenuniv.nl>

  if keyword_set(help) then begin
     print
     print, 'result = compute_cooling(tablepath, tabletype, redshift, density, temperature, AH, AHe, AC, AN, AO, ANe, AMg, ASi, AS, ACa, AFe, /help)'
     print
     print, 'This routine calculates the net cooling rate (Lambda/n_H^2 [erg s^-1 cm^3]) given an array of redshifts, densities, temperatures, and abundnces using the cooling tables prepared by Wiersma, Schaye, & Smith (2009). Arrays or single values may be used provided the dimensions are consistent.'
     print
     print, 'result: an array of net cooling rates, in (Lambda_net/n_H^2 [erg cm^3 s^-1])'
     print, 'tablepath: the path containing the coolingtables'
     print, 'tabletype can be:' 
     print, '   normal: interpolate in redshift using the tables'
     print, '   collis: calculate the cooling for collisional ionization equilibrium'
     print, '   photodis: calculate the cooling for a soft, non-ionizing UV background. It uses the z = 8.989 HM01 background but with the spectrum cut-off at 1 Ryd. Compton cooling off the CMB is added based on redshift (use a negative redshift value to exclude it). This option could be used in the pre-reionization universe.'
     print, '   highz: calculate the cooling using the  z = 8.989 HM01 background and include compton cooling off the cmb based on redshift (use a negative redshift value to exclude it). This option could be used for 9 < z < z_reion.'
     print, 'redshift: the redshift (for collisional ionization, enter a dummy value)'
     print, 'density: the hydrogen number density (n_H [cm^-3])'
     print, 'temperature: the temperature (T [K])'
     print, 'AX: abundance of element X. Can be mass fractions (if massfrac = 1) or number densities relative to hydrogen (if massfrac = 0). Note that number densities should give a slightly more accurate result'
     print, 'massfrac can be:'
     print, '   0: Abundance is number density relative to H (more accurate method)'
     print, '   1: Abundance is mass fraction (less accurate method)'
     print
     return, 0
  endif

  STEFAN = 7.5657e-15 ;; erg cm^-3 K^-4
  C_speed = 2.9979e10 ;; cm s^-1
  ELECTRONMASS = 9.10953e-28 ;; g
  THOMPSON = 6.6524587e-25 ;; cm^2
  BOLTZMANN = 1.3806e-16 ;; erg K^-1

  ;; Error handling
  if n_params() ne 17 then message,'Wrong number of parameters! Try /help option for more information'

  nz = n_elements(redshift)
  if nz ne n_elements(density) or nz ne n_elements(temperature) or $
     nz ne n_elements(AH) or nz ne n_elements(AHe) or nz ne n_elements(AC) or $
     nz ne n_elements(AO) or nz ne n_elements(ANe) or nz ne n_elements(AMg) or $
     nz ne n_elements(ASi) or nz ne n_elements(AS) or nz ne n_elements(ACa) or $
     nz ne n_elements(AFe) then begin
     print,nz,n_elements(density),n_elements(temperature), $
           n_elements(AH), n_elements(AHe), n_elements(AC), $
           n_elements(AN), n_elements(AO), n_elements(ANe), $
           n_elements(AMg), n_elements(ASi), n_elements(AS), $
           n_elements(ACa), n_elements(AFe)
     message,'All input arrays must have equal number of elements!' 
  endif
  case massfrac of
     0: string = 'number densities'
     1: string = 'mass fractions'
     else: message,'massfrac must be either zero or one!'
  endcase

  if tabletype ne 'normal' and tabletype ne 'photodis' and $
     tabletype ne 'highz' and tabletype ne 'collis' then $
        message,tabletype + ' = unknown table type!'

  ;; Print info
  print,format='("Computing cooling for ",g8.3," elements")',nz
  print,format='("Assuming helium abundances are ",a)',string
  print,format='("Using table path ",a)',tablepath
  print,format='("Using table type ",a)',tabletype

  ;; Check of all redshifts are the same (if so, we can speed things up)
  nuniqz = n_elements(uniq(redshift,sort(redshift)))

  net_cool = fltarr(nz)

  if tabletype eq 'normal' then begin
     ;;;; normal tables ;;;;

     ;; first fill the redshift array
     files = file_search(tablepath+'/*.hdf5')
     if n_elements(files) eq 1 and files[0] eq '' then $
        message,'No .hdf5 files found in directory ' + tablepath
     table_redshifts = fltarr(n_elements(files) -3)
     table_indexes = fltarr(n_elements(files) -3)
     j = 0
     for i = 0, n_elements(files) - 1 do begin
        if strpos(files[i],'collis') eq -1 and strpos(files[i],'compt') eq -1 $
           and strpos(files[i],'photo') eq -1 then begin 
           table_redshifts[j] = $
              strmid(files[i], strpos(files[i], '_',/reverse_search)+1, $
                     strpos(files[i],'.', /reverse_search)- $
                     strpos(files[i], '_',/reverse_search)-1)
           table_indexes[j] = i
           j = j+1
        endif
     endfor

     z_ind = intarr(nz)
     dz = fltarr(nz)
     for i = 0, nz -1 do begin
        ;; find the redshift index
        z_ind[i] = min(where(table_redshifts ge redshift[i]))
        if z_ind[i] eq 0 then begin ;; bottom of the table
           z_ind[i] = 1
           dz[i] = 1.
        endif else if z_ind[i] eq -1 then begin ;; top of the table
           z_ind[i] = n_elements(table_redshifts) -2
           dz[i] = 0.
        endif else dz[i] = (table_redshifts[z_ind[i]] - redshift[i]) / $
                           (table_redshifts[z_ind[i]] - $
                            table_redshifts[z_ind[i]-1])

        if nuniqz eq 1 then begin
           z_ind[*] = z_ind[0]
           dz[*] = dz[0]
           break
        endif

     endfor
     if min(where(redshift lt min(table_redshifts))) ne -1 or $
        min(where(redshift gt max(table_redshifts))) ne -1 then begin
        print, 'There are one or more points outside of the redshift range! Table end points will be used'
        print, 'Type any key to continue.'
        WHILE Get_KBRD(0) do junk=1
        junk = Get_KBRD(1)
        WHILE Get_KBRD(0) do junk=1
     endif

     data1 = h5_parse(files[table_indexes[z_ind[0] - 1]], /read_data)
     data2 = h5_parse(files[table_indexes[z_ind[0]]], /read_data)

     hhe_cool1 = data1.metal_free.net_cooling._data
     hhe_ne1 =data1.metal_free.Electron_density_over_n_h._data
     hhe_cool2 = data2.metal_free.net_cooling._data
     hhe_ne2 =data2.metal_free.Electron_density_over_n_h._data
     solar_ne_nh1 = data1.solar.electron_density_over_n_h._data
     solar_ne_nh2 = data2.solar.electron_density_over_n_h._data
     carb_cool1 = data1.Carbon.net_cooling._data 
     nitr_cool1 = data1.Nitrogen.net_cooling._data 
     oxyg_cool1 = data1.Oxygen.net_cooling._data
     neon_cool1 = data1.Neon.net_cooling._data
     magn_cool1 = data1.Magnesium.net_cooling._data
     sili_cool1 = data1.Silicon.net_cooling._data 
     sulp_cool1 = data1.Sulphur.net_cooling._data
     calc_cool1 = data1.Calcium.net_cooling._data
     iron_cool1 = data1.Iron.net_cooling._data
     
     carb_cool2 = data2.Carbon.net_cooling._data 
     nitr_cool2 = data2.Nitrogen.net_cooling._data 
     oxyg_cool2 = data2.Oxygen.net_cooling._data
     neon_cool2 = data2.Neon.net_cooling._data
     magn_cool2 = data2.Magnesium.net_cooling._data
     sili_cool2 = data2.Silicon.net_cooling._data 
     sulp_cool2 = data2.Sulphur.net_cooling._data
     calc_cool2 = data2.Calcium.net_cooling._data
     iron_cool2 = data2.Iron.net_cooling._data

     tbl_hhecool = hhe_cool1 * dz[0] + hhe_cool2 * (1. - dz[0])
     tbl_hhene = hhe_ne1 * dz[0] + hhe_ne2 * (1. - dz[0])
     tbl_solar_ne_nh = solar_ne_nh1 * dz[0] + solar_ne_nh2 * (1. - dz[0])
     tbl_carbcool = carb_cool1 * dz[0] + carb_cool2 * (1. - dz[0])
     tbl_nitrcool = nitr_cool1 * dz[0] + nitr_cool2 * (1. - dz[0])
     tbl_oxygcool = oxyg_cool1 * dz[0] + oxyg_cool2 * (1. - dz[0])
     tbl_neoncool = neon_cool1 * dz[0] + neon_cool2 * (1. - dz[0])
     tbl_magncool = magn_cool1 * dz[0] + magn_cool2 * (1. - dz[0])
     tbl_silicool = sili_cool1 * dz[0] + sili_cool2 * (1. - dz[0])
     tbl_sulpcool = sulp_cool1 * dz[0] + sulp_cool2 * (1. - dz[0])
     tbl_calccool = calc_cool1 * dz[0] + calc_cool2 * (1. - dz[0])
     tbl_ironcool = iron_cool1 * dz[0] + iron_cool2 * (1. - dz[0])
     
  endif else begin
     case tabletype of
        'collis': file = 'z_collis'
        'photodis': file = 'z_photodis'
        'highz': file = 'z_8.989nocompton'
     endcase
     data1 = h5_parse(tablepath+'/' + file + '.hdf5', /read_data)

     tbl_hhecool = data1.metal_free.net_cooling._data
     tbl_hhene = data1.metal_free.Electron_density_over_n_h._data
     tbl_solar_ne_nh = data1.solar.electron_density_over_n_h._data
     tbl_carbcool = data1.Carbon.net_cooling._data
     tbl_nitrcool = data1.Nitrogen.net_cooling._data
     tbl_oxygcool = data1.Oxygen.net_cooling._data
     tbl_neoncool = data1.Neon.net_cooling._data
     tbl_magncool = data1.Magnesium.net_cooling._data
     tbl_silicool = data1.Silicon.net_cooling._data
     tbl_sulpcool = data1.Sulphur.net_cooling._data
     tbl_calccool = data1.Calcium.net_cooling._data
     tbl_ironcool = data1.Iron.net_cooling._data
  endelse
        
  if tabletype ne 'collis' then begin
     tbl_log_dens = alog10(data1.metal_free.hydrogen_density_bins._data)

     dens_ind = (n_elements(tbl_log_dens) - 1.) * $
                (alog10(density) - min(tbl_log_dens)) / $
                (max(tbl_log_dens) - min(tbl_log_dens))

     if min(where(alog10(density) lt min(tbl_log_dens))) ne -1 or $
            min(where(alog10(density) gt max(tbl_log_dens))) ne -1 then begin
        print, 'There are one or more points outside of the density range! Table end points will be used'
        print, 'Type any key to continue.'
        WHILE Get_KBRD(0) do junk=1
        junk = Get_KBRD(1)
        WHILE Get_KBRD(0) do junk=1
     endif
  endif

  tbl_log_temperature = alog10(data1.metal_free.temperature_bins._data)
     
  temperature_ind = (n_elements(tbl_log_temperature) - 1.) * $
                    (alog10(temperature) - min(tbl_log_temperature)) / $
                    (max(tbl_log_temperature) - min(tbl_log_temperature))
                        
  if min(where(alog10(temperature) lt min(tbl_log_temperature))) ne -1 or $
         min(where(alog10(temperature) gt max(tbl_log_temperature))) ne -1 then begin
     print, 'There are one or more points outside of the temperature range! Table end points will be used'
     print, 'Type any key to continue.'
     WHILE Get_KBRD(0) do junk=1
     junk = Get_KBRD(1)
     WHILE Get_KBRD(0) do junk=1
  endif

  if massfrac eq 1 then begin
     tbl_AHe = data1.Metal_free.Helium_mass_fraction_bins._data 
     hhe_frac = AHe / (AH + AHe)
     AHe_ind = (n_elements(tbl_AHe) - 1.) * (hhe_frac - min(tbl_AHe)) / $
               (max(tbl_AHe) - min(tbl_AHe))

     solar_abund = data1.Header.Abundances.Solar_mass_fractions._data
  endif else begin
     tbl_AHe = data1.Metal_free.Helium_number_ratio_bins._data
     hhe_frac = AHe / AH 
     ;; note that in number density space, the tables are not evenly spaced!
     ;; Check of all helium fracs are the same (if so, we can speed things up)
     nuniqhe = n_elements(uniq(hhe_frac,sort(hhe_frac)))

     AHe_ind = fltarr(nz)
     for i = 0, nz-1 do begin
        AHe_ind[i] = min(where(tbl_AHe ge hhe_frac[i]))
        if AHe_ind[i] eq -1 then $ ;; top of the table
           AHe_ind[i] = n_elements(tbl_AHe)-1 $
        else if AHe_ind[i] ne 0 then $
           AHe_ind[i] = AHe_ind[i] - $
                        ((tbl_AHe[AHe_ind[i]] - hhe_frac[i])/ $
                         (tbl_AHe[AHe_ind[i]] - tbl_AHe[AHe_ind[i]-1]))
        if nuniqhe eq 1 then begin
           AHe_ind[*] = AHe_ind[0]
           break
        endif

     endfor

     solar_abund = data1.Header.Abundances.Solar_number_ratios._data
  endelse
  if min(where(hhe_frac lt min(tbl_AHe))) ne -1 or $
         min(where(hhe_frac gt max(tbl_AHe))) ne -1 then begin
     print, 'There are one or more points outside of the Helium range! Table end points will be used'
     print, 'Type any key to continue.'
     WHILE Get_KBRD(0) do junk=1
     junk = Get_KBRD(1)
     WHILE Get_KBRD(0) do junk=1
  endif

  for i = 0, nz-1 do begin
     ;; check if data needs to be read and if z-interpolation needs to be done
     if tabletype eq 'normal' then begin
        if i gt 0 then begin
           if z_ind[i] ne z_ind[i-1] then need_to_read = 1 $
           else need_to_read = 0
           if redshift[i] ne redshift[i-1] then need_to_interpolate_z = 1 $
           else need_to_interpolate_z = 0
        endif else begin
           need_to_read = 1
           need_to_interpolate_z = 1
        endelse
        ;; read data if necessary
        if need_to_read then begin
           data1 = h5_parse(files[table_indexes[z_ind[i]-1]], /read_data)
           data2 = h5_parse(files[table_indexes[z_ind[i]]], /read_data)

           hhe_cool1 = data1.metal_free.net_cooling._data
           hhe_ne1 =data1.metal_free.Electron_density_over_n_h._data
           hhe_cool2 = data2.metal_free.net_cooling._data
           hhe_ne2 =data2.metal_free.Electron_density_over_n_h._data
           solar_ne_nh1 = data1.solar.electron_density_over_n_h._data
           solar_ne_nh2 = data2.solar.electron_density_over_n_h._data
           carb_cool1 = data1.Carbon.net_cooling._data 
           nitr_cool1 = data1.Nitrogen.net_cooling._data 
           oxyg_cool1 = data1.Oxygen.net_cooling._data
           neon_cool1 = data1.Neon.net_cooling._data
           magn_cool1 = data1.Magnesium.net_cooling._data
           sili_cool1 = data1.Silicon.net_cooling._data 
           sulp_cool1 = data1.Sulphur.net_cooling._data
           calc_cool1 = data1.Calcium.net_cooling._data
           iron_cool1 = data1.Iron.net_cooling._data

           carb_cool2 = data2.Carbon.net_cooling._data 
           nitr_cool2 = data2.Nitrogen.net_cooling._data 
           oxyg_cool2 = data2.Oxygen.net_cooling._data
           neon_cool2 = data2.Neon.net_cooling._data
           magn_cool2 = data2.Magnesium.net_cooling._data
           sili_cool2 = data2.Silicon.net_cooling._data 
           sulp_cool2 = data2.Sulphur.net_cooling._data
           calc_cool2 = data2.Calcium.net_cooling._data
           iron_cool2 = data2.Iron.net_cooling._data
        endif
        
        ;; interpolate z if necessary
        if need_to_interpolate_z then begin
           tbl_hhecool = hhe_cool1 * dz[i] + hhe_cool2 * (1. - dz[i])
           tbl_hhene = hhe_ne1 * dz[i] + hhe_ne2 * (1. - dz[i])
           tbl_solar_ne_nh = solar_ne_nh1 * dz[i] + solar_ne_nh2 * (1. - dz[i])
           tbl_carbcool = carb_cool1 * dz[i] + carb_cool2 * (1. - dz[i])
           tbl_nitrcool = nitr_cool1 * dz[i] + nitr_cool2 * (1. - dz[i])
           tbl_oxygcool = oxyg_cool1 * dz[i] + oxyg_cool2 * (1. - dz[i])
           tbl_neoncool = neon_cool1 * dz[i] + neon_cool2 * (1. - dz[i])
           tbl_magncool = magn_cool1 * dz[i] + magn_cool2 * (1. - dz[i])
           tbl_silicool = sili_cool1 * dz[i] + sili_cool2 * (1. - dz[i])
           tbl_sulpcool = sulp_cool1 * dz[i] + sulp_cool2 * (1. - dz[i])
           tbl_calccool = calc_cool1 * dz[i] + calc_cool2 * (1. - dz[i])
           tbl_ironcool = iron_cool1 * dz[i] + iron_cool2 * (1. - dz[i])
        endif
        hhe_cool = interpolate(tbl_hhecool,dens_ind[i],temperature_ind[i], $
                               AHe_ind[i])
        hhe_ne = interpolate(tbl_hhene,dens_ind[i],temperature_ind[i], $
                             AHe_ind[i])
        solar_ne = bilinear(tbl_solar_ne_nh,dens_ind[i],temperature_ind[i])

        metal_cool = tbl_carbcool * (AC[i]/solar_abund[2]) + $
                     tbl_nitrcool * (AN[i]/solar_abund[3]) + $
                     tbl_oxygcool * (AO[i]/solar_abund[4]) + $
                     tbl_neoncool * (ANe[i]/solar_abund[5]) + $
                     tbl_magncool * (AMg[i]/solar_abund[6]) + $
                     tbl_silicool * (ASi[i]/solar_abund[7]) + $
                     tbl_sulpcool * (AS[i]/solar_abund[8]) + $
                     tbl_calccool * (ACa[i]/solar_abund[9]) + $
                     tbl_ironcool * (AFe[i]/solar_abund[10])
        
        net_cool[i] = (hhe_cool + (hhe_ne/solar_ne) * $
                       bilinear(metal_cool,dens_ind[i],temperature_ind[i]))
     endif else if tabletype ne 'collis' then begin
        hhe_cool = interpolate(tbl_hhecool,dens_ind[i],temperature_ind[i], $
                               AHe_ind[i])
        hhe_ne = interpolate(tbl_hhene,dens_ind[i],temperature_ind[i], $
                             AHe_ind[i])
        solar_ne = bilinear(tbl_solar_ne_nh,dens_ind[i],temperature_ind[i])

        metal_cool = tbl_carbcool * (AC[i]/solar_abund[2]) + $
                     tbl_nitrcool * (AN[i]/solar_abund[3]) + $
                     tbl_oxygcool * (AO[i]/solar_abund[4]) + $
                     tbl_neoncool * (ANe[i]/solar_abund[5]) + $
                     tbl_magncool * (AMg[i]/solar_abund[6]) + $
                     tbl_silicool * (ASi[i]/solar_abund[7]) + $
                     tbl_sulpcool * (AS[i]/solar_abund[8]) + $
                     tbl_calccool * (ACa[i]/solar_abund[9]) + $
                     tbl_ironcool * (AFe[i]/solar_abund[10])

        ; Add Compton cooling off the CMB
        if redshift[i] ge 0. then begin
           t_cmb = 2.728 * (1.0 + redshift[i])          
           comp_add = - (4.0 * STEFAN * THOMPSON * (t_cmb^4) / $
                         (ELECTRONMASS * C_speed)) * BOLTZMANN * $
                      (t_cmb - temperature[i]) * hhe_ne / density[i] 
        endif else comp_add = 0.
        
        net_cool[i] = (hhe_cool + (hhe_ne/solar_ne) * $
                       bilinear(metal_cool,dens_ind[i], temperature_ind[i]) $
                       + comp_add)
     endif else begin
        hhe_cool = bilinear(tbl_hhecool,temperature_ind[i],AHe_ind[i])
        hhe_ne = bilinear(tbl_hhene,temperature_ind[i],AHe_ind[i])
        solar_ne = interpol(tbl_solar_ne_nh,tbl_log_temperature, $
                            alog10(temperature[i]))

        metal_cool = tbl_carbcool * (AC[i]/solar_abund[2]) + $
                     tbl_nitrcool * (AN[i]/solar_abund[3]) + $
                     tbl_oxygcool * (AO[i]/solar_abund[4]) + $
                     tbl_neoncool * (ANe[i]/solar_abund[5]) + $
                     tbl_magncool * (AMg[i]/solar_abund[6]) + $
                     tbl_silicool * (ASi[i]/solar_abund[7]) + $
                     tbl_sulpcool * (AS[i]/solar_abund[8]) + $
                     tbl_calccool * (ACa[i]/solar_abund[9]) + $
                     tbl_ironcool * (AFe[i]/solar_abund[10])
        
        net_cool[i] = (hhe_cool + (hhe_ne/solar_ne) * $
                       interpol(metal_cool,tbl_log_temperature, $
                                alog10(temperature[i])))
     endelse
        
  endfor


  return, net_cool
end
