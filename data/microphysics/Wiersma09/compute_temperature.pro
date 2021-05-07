function compute_temperature, $
   tablepath,tabletype,redshift,density,energy_density,AH,AHe,massfrac, $
   help=help

; This routine converts energy density to temperature. See
; test = compute_temperature(/help)
; for more details

; This is version 100423.
; Change history:
;      * 08????: Original version
;      * 090512: Fixed issue regarding use of uniform redshift arrays
;                (the code used to crash). Updated reference.
;      * 100423: Clarified the help.
;
; Please refer to Wiersma, Schaye, & Smith 2009, MNRAS, 393, 99
; for a description of the methods used.
; 
; For bug reports and questions, please contact 
; Rob Wiersma <wiersma@strw.leidenuniv.nl>

  if keyword_set(help) then begin
     print
     print, 'result = compute_temperature(tablepath, tabletype, redshift, density, energy_density, AH, AHe, massfrac, /help)'
     print
     print, 'This routine converts energy density to temperature given an array of'
     print, 'redshifts, densities, energy densities, and helium fractions using'
     print, 'the tables prepared by Wiersma, Schaye, & Smith (2009).'
     print, 'Arrays or single values may be used provided the dimensions are consistent.'
     print
     print, 'result: an array of temperatures (T [K])'
     print, 'tablepath: the path containing the coolingtables'
     print, 'tabletype can be:' 
     print, '   normal: interpolate in redshift using the tables'
     print, '   collis: calculate the temperature for collisional ionization equilibrium'
     print, '   photodis: calculate the temperature for a soft, non-ionizing UV background. It uses the z = 8.989 HM01 background but with the spectrum cut-off at 1 Ryd. This option could be used in the pre-reionization universe.'
     print, '   highz: calculate the temperature using the  z = 8.989 HM01 background. This option could be used for 9 < z < z_reion.'
     print, 'redshift: the redshift (for collisional ionization, enter dummy values)'
     print, 'density: the hydrogen number density (n_H [cm^-3])'
     print, 'energy_desity: the energy density per unit mass (u [erg g^-1])'
     print, 'AH: abundance of Hydrogen' 
     print, 'AHe: abundance of Helium' 
     print, 'massfrac can be:'
     print, '   0: Abundance is number density relative to H (more accurate method)'
     print, '   1: Abundance is mass fraction (less accurate method)'
     print
     return,0
  endif

  ;; Error handling
  if n_params() ne 8 then message,'Wrong number of parameters! Try /help option for more information.'

  nz = n_elements(redshift)
  if nz ne n_elements(density) or nz ne n_elements(energy_density) or $
     nz ne n_elements(AH) or nz ne n_elements(AHe) then begin
     print,nz,n_elements(density),n_elements(energy_density), $
           n_elements(AH),n_elements(AHe)
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
  print,format='("Computing temperatures for ",g8.3," elements")',nz
  print,format='("Assuming helium abundances are ",a)',string
  print,format='("Using table path ",a)',tablepath
  print,format='("Using table type ",a)',tabletype

  ;; Check whether all redshifts are the same (if so, we can speed things up)
  nuniqz = n_elements(uniq(redshift,sort(redshift)))

  if tabletype eq 'normal' then begin
     ;; first fill the redshift array
     files = file_search(tablepath+'/*.hdf5')
     if n_elements(files) eq 1 and files[0] eq '' then $
        message,'No .hdf5 files found in directory ' + tablepath
     table_redshifts = fltarr(n_elements(files) -3)
     table_indexes = fltarr(n_elements(files) -3)
     j = 0
     for i = 0, n_elements(files) - 1 do begin
        if strpos(files[i],'collis') eq -1 and $
           strpos(files[i],'compt') eq -1 and $
           strpos(files[i],'photo') eq -1 then begin 
           table_redshifts[j] = strmid(files[i], strpos(files[i], '_',/reverse_search)+1, strpos(files[i], '.', /reverse_search)-strpos(files[i], '_',/reverse_search)-1)
           table_indexes[j] = i
           j = j+1
        endif
     endfor

     z_ind = intarr(nz)
     dz = fltarr(nz)
     for i = 0, nz-1 do begin
        z_ind[i] = min(where(table_redshifts ge redshift[i]))
        if z_ind[i] eq 0 then begin ;; bottom of the table
           z_ind[i] = 1
           dz[i] = 1.
        endif else if z_ind[i] eq -1 then begin ;; top of the table
           z_ind[i] = n_elements(table_redshifts)-2
           dz[i] = 0.
        endif else $
           dz[i] = (table_redshifts[z_ind[i]] - redshift[i]) / $
                   (table_redshifts[z_ind[i]] - table_redshifts[z_ind[i]-1])
        if nuniqz eq 1 then begin
           z_ind[*] = z_ind[0]
           dz[*] = dz[0]
           break
        endif
     endfor

     ;; read data
     data1 = h5_parse(files[table_indexes[z_ind[0]-1]], /read_data)
     data2 = h5_parse(files[table_indexes[z_ind[0]]], /read_data)
     
     tbl1_temp = data1.metal_free.temperature.temperature._data
     tbl2_temp = data2.metal_free.temperature.temperature._data
  endif else begin
     case tabletype of
        'collis': file = 'z_collis'
        'photodis': file = 'z_photodis'
        'highz': file = 'z_8.989nocompton'
     endcase
     data1 = h5_parse(tablepath+'/' + file + '.hdf5', /read_data)

     tbl1_temp = data1.metal_free.temperature.temperature._data
  endelse
  
  
  if tabletype ne 'collis' then begin
     tbl_log_dens = alog10(data1.metal_free.hydrogen_density_bins._data)

     if min(where(alog10(density) lt min(tbl_log_dens))) ne -1 or $
            min(where(alog10(density) gt max(tbl_log_dens))) ne -1 then begin
        print, 'There are one or more points outside of the density range! Table end points will be used'
        print, 'Type any key to continue.'
        read, dummy
     endif

     dens_ind = (n_elements(tbl_log_dens) - 1.) * $
                (alog10(density) - min(tbl_log_dens)) / $
                (max(tbl_log_dens) - min(tbl_log_dens))
  endif

  tbl_log_energy_dens = alog10(data1.metal_free.temperature.energy_density_bins._data)
     
  if min(where(alog10(energy_density) lt min(tbl_log_energy_dens))) ne -1 or $
         min(where(alog10(energy_density) gt max(tbl_log_energy_dens))) ne -1 then begin
     print, 'There are one or more points outside of the energy density range! Values will be extrapolated'
     print, 'Type any key to continue.'
     read, dummy
  endif

  energy_dens_ind = (n_elements(tbl_log_energy_dens) - 1.) * $
                    (alog10(energy_density) - min(tbl_log_energy_dens)) / $
                    (max(tbl_log_energy_dens) - min(tbl_log_energy_dens))
                        
  if massfrac eq 1 then begin
     hhe_frac = AHe / (AH + AHe)

     tbl_AHe = data1.Metal_free.Helium_mass_fraction_bins._data 
     AHe_ind = (n_elements(tbl_AHe) - 1.) * (hhe_frac - min(tbl_AHe)) / $
               (max(tbl_AHe) - min(tbl_AHe))
  endif else begin
     tbl_AHe = data1.Metal_free.Helium_number_ratio_bins._data
     ;; note that in number density space, the tables are not evenly spaced!
     ;; Check if all helium fracs are the same (if so, we can speed things up)
     nuniqhe = n_elements(uniq(AHe,sort(AHe)))

     hhe_frac = AHe / AH 
     AHe_ind = fltarr(nz)
     for i = 0, nz-1 do begin
        AHe_ind[i] = min(where(tbl_AHe ge hhe_frac[i]))
        if AHe_ind[i] eq -1 then $ ;; top of the table
           AHe_ind[i] = n_elements(tbl_AHe)-1 $
        else if AHe_ind[i] ne 0 then $
           AHe_ind[i] = AHe_ind[i] - $
                        ((tbl_AHe[AHe_ind[i]] - hhe_frac[i]) / $
                         (tbl_AHe[AHe_ind[i]] - tbl_AHe[AHe_ind[i]-1]))
        if nuniqhe eq 1 then begin
           AHe_ind[*] = AHe_ind[0]
           break
        endif

     endfor
  endelse
  if min(where(hhe_frac lt min(tbl_AHe))) ne -1 or $
         min(where(hhe_frac gt max(tbl_AHe))) ne -1 then begin
     print, 'There are one or more points outside of the Helium range! Table end points will be used'
     print, 'Type any key to continue.'
     read, dummy
  endif

  log_temp = fltarr(nz)
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
           
           tbl1_temp = data1.metal_free.temperature.temperature._data
           tbl2_temp = data2.metal_free.temperature.temperature._data
        endif
        
        ;; interpolate z if necessary
        if need_to_interpolate_z then $
           tbl_log_temp = alog10(tbl1_temp) * dz[i] + $
                          alog10(tbl2_temp) * (1. - dz[i])
     endif else $
        tbl_log_temp = alog10(tbl1_temp)

     if tabletype ne 'collis' then begin
        ;; interpolate in density, energy density, and helium abundance
        if n_elements(energy_dens_ind) eq 1 then begin
           if energy_dens_ind lt 0 then begin
              log_temp1 = interpolate(tbl_log_temp,dens_ind, $
                                        0,AHe_ind)
              log_temp2 = interpolate(tbl_log_temp,dens_ind, $
                                        1,AHe_ind)
              log_temp = ((log_temp2 - log_temp1) / (tbl_log_energy_dens[1] - tbl_log_energy_dens[0])) * $
                         (alog10(energy_density) - tbl_log_energy_dens[0]) + log_temp1
           endif else if energy_dens_ind gt n_elements(tbl_log_energy_dens) - 1 then begin 
              log_temp1 = interpolate(tbl_log_temp,dens_ind, $
                                        n_elements(tbl_log_energy_dens) - 2,AHe_ind)
              log_temp2 = interpolate(tbl_log_temp,dens_ind, $
                                        n_elements(tbl_log_energy_dens) - 1,AHe_ind)
              log_temp = ((log_temp2 - log_temp1) / (tbl_log_energy_dens[1] - tbl_log_energy_dens[0])) * $
                         (alog10(energy_density) - tbl_log_energy_dens[0]) + log_temp1              
           endif else $         
                 log_temp = interpolate(tbl_log_temp,dens_ind, $
                                        energy_dens_ind,AHe_ind)
           return,10.^log_temp
        endif else begin
           if energy_dens_ind[i] lt 0 then begin
              log_temp1 = interpolate(tbl_log_temp,dens_ind[i], $
                                        0,AHe_ind[i])
              log_temp2 = interpolate(tbl_log_temp,dens_ind[i], $
                                        1,AHe_ind[i])
              log_temp[i] = ((log_temp2 - log_temp1) / (tbl_log_energy_dens[1] - tbl_log_energy_dens[0])) * $
                         (alog10(energy_density[i]) - tbl_log_energy_dens[0]) + log_temp1
           endif else if energy_dens_ind[i] gt n_elements(tbl_log_energy_dens) - 1 then begin 
              log_temp1 = interpolate(tbl_log_temp,dens_ind[i], $
                                        n_elements(tbl_log_energy_dens) - 2,AHe_ind[i])
              log_temp2 = interpolate(tbl_log_temp,dens_ind[i], $
                                        n_elements(tbl_log_energy_dens) - 1,AHe_ind[i])
              log_temp[i] = ((log_temp2 - log_temp1) / (tbl_log_energy_dens[1] - tbl_log_energy_dens[0])) * $
                         (alog10(energy_density[i]) - tbl_log_energy_dens[0]) + log_temp1              
           endif else $
              log_temp[i] = interpolate(tbl_log_temp,dens_ind[i], $
                                        energy_dens_ind[i],AHe_ind[i])
        endelse
     endif else begin
        ;; interpolate in energy density, and helium abundance
        if energy_dens_ind[i] lt 0 then begin
           log_temp1 = interpolate(tbl_log_temp,0,AHe_ind)
           log_temp2 = interpolate(tbl_log_temp,1,AHe_ind)
           log_temp = ((log_temp2 - log_temp1) / (tbl_log_energy_dens[1] - tbl_log_energy_dens[0])) * $
                      (alog10(energy_density) - tbl_log_energy_dens[0]) + log_temp1
        endif else if energy_dens_ind gt n_elements(tbl_log_energy_dens) - 1 then begin 
           log_temp1 = interpolate(tbl_log_temp, $
                                   n_elements(tbl_log_energy_dens) - 2,AHe_ind)
           log_temp2 = interpolate(tbl_log_temp, $
                                   n_elements(tbl_log_energy_dens) - 1,AHe_ind)
           log_temp = ((log_temp2 - log_temp1) / (tbl_log_energy_dens[1] - tbl_log_energy_dens[0])) * $
                      (alog10(energy_density) - tbl_log_energy_dens[0]) + log_temp1              
        endif else $
           log_temp = interpolate(tbl_log_temp,energy_dens_ind,AHe_ind)
        return, 10.^log_temp
     endelse
  endfor
  
  return,10.^log_temp

end
