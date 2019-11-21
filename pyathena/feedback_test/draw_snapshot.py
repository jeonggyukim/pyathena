
def draw_snapshot(s, nums, view=True):
    u = pa.Units(kind='LV')
    h = pa.read_hst(s.files['hst'])
    if s.par['configure']['new_cooling'] == 'OFF':
        newcool = False
    else:
        newcool = True
        
    for num in nums:
        print(num, end=' ')
        ds = s.load_vtk(num)
        
        from matplotlib import gridspec
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(nrows=3, ncols=3, wspace=0.5,
                               height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[0,2])
        ax3 = fig.add_subplot(gs[1,0])
        ax4 = fig.add_subplot(gs[1,1])
        ax5 = fig.add_subplot(gs[1,2])
        ax6 = fig.add_subplot(gs[2,0])
        ax7 = fig.add_subplot(gs[2,1])
        ax8 = fig.add_subplot(gs[2,2])
        axes = (ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8)
        
        # Plane through which to make slices
        slc = dict(y=0, method='nearest')
        
        # Read data and calculate derived fields
        if newcool:
            d = ds.get_field(['nH', 'nH2', 'temperature', 'pok', 'heat_ratio', 'velocity',
                              'rad_energy_density_PE', 'rad_energy_density_PE_unatt'])
        else:
            d = ds.get_field(['nH', 'temperature', 'pok', 'heat_ratio', 'velocity',
                              'rad_energy_density_PE', 'rad_energy_density_PE_unatt'])
            
        d['AVeff'] = -1.87*np.log(d['rad_energy_density_PE']/d['rad_energy_density_PE_unatt'])
        d['r'] = np.sqrt((d.x**2 + d.y**2 + d.z**2))
        d['vr'] = np.sqrt(d['velocity1']**2+d['velocity2']**2+d['velocity3']**2)

        # Plot slices
        d['nH'].sel(**slc).plot.imshow(ax=ax0, norm=LogNorm(1e-5,1e4), cmap=mpl.cm.Spectral_r)
        d['temperature'].sel(**slc).plot.imshow(ax=ax1, norm=LogNorm(1e1,1e7),
                                   cmap=pa.cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7.))
        d['pok'].sel(**slc).plot.imshow(ax=ax3, norm=LogNorm(1e1,1e7), cmap=mpl.cm.jet)
        d['heat_ratio'].sel(**slc).plot.imshow(ax=ax4, norm=LogNorm(1e-1,1e4), cmap=mpl.cm.viridis)
        d['vr'].sel(**slc).plot.imshow(ax=ax6, norm=Normalize(-10, 10.0), cmap=mpl.cm.PiYG)
        
        plt.sca(ax2)
        h.plot(x='time', y=['rmom_bub','Rsh', 'Minter', 'Mwarm', 'Mhot', 'Mcold'],
               logy=True, marker='o', markersize=2, ax=ax2)
        plt.ylim(plt.gca().get_ylim()[1]*1e-4, plt.gca().get_ylim()[1])
        plt.axvline(ds.domain['time'], color='grey', linestyle='--')

        plt.sca(ax5)
        plt.hist2d(d['nH'].data.flatten(), d['pok'].data.flatten(),
                   bins=(np.logspace(-3,4,100), np.logspace(1,7,100)), norm=LogNorm());
        cf = pa.classic.cooling.coolftn()
        plt.plot(cf.heat/cf.cool,
                 1.1*cf.get_temp(cf.T1)*cf.heat/cf.cool, c='r', ls=':',label='KI02')
        plt.plot(10.0*cf.heat/cf.cool,
                 1.1*cf.get_temp(cf.T1)*10.0*cf.heat/cf.cool, c='r', ls=':',label='KI02')
        plt.plot(100.0*cf.heat/cf.cool,
                 1.1*cf.get_temp(cf.T1)*100.0*cf.heat/cf.cool, c='r', ls=':',label='KI02')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('nH')
        plt.ylabel('pok')

        # stride for scatter plot
        j = 10
        plt.sca(ax7)
        plt.scatter(d['r'].data.flatten()[::j], d['nH'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C0', label='nH')
        plt.scatter(d['r'].data.flatten()[::j], 1e1*d['temperature'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C1', label='T*10')
        plt.scatter(d['r'].data.flatten()[::j], d['vr'].data.flatten()[::j],
                        marker='o', s=1.0, alpha=1, c='C2', label='vr')
        plt.scatter(d['r'].data.flatten()[::j], d['AVeff'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C3', label='AVeff')
        plt.yscale('log')
        plt.ylim(1e-3, 1e5)
        plt.xlabel('r')
        plt.ylabel('nH, nH2, T*10, AVeff')
        plt.legend(loc=1)

        plt.sca(ax8)
        plt.scatter(d['AVeff'].data.flatten()[::j], d['nH'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C0', label='nH')
        if newcool:
            plt.scatter(d['AVeff'].data.flatten()[::j], d['nH2'].data.flatten()[::j],
                        marker='o', s=1.0, alpha=1, c='C2', label='nH2')
        plt.scatter(d['AVeff'].data.flatten()[::j], 1e1*d['temperature'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C5', label='T*10')
        plt.scatter(d['AVeff'].data.flatten()[::j], d['heat_ratio'].data.flatten()[::j],
                    marker='o', s=1.0, alpha=1, c='C4', label='heat_ratio')
        plt.yscale('log')
        plt.xlim(0, 15)
        plt.ylim(1e-2, 1e5)
        plt.xlabel('Aveff')
        plt.ylabel('nH, nH2, T*10, heat_ratio')
        plt.legend(loc=1)
        
        for ax in (ax0,ax1,ax3,ax4,ax6):
            ax.set_aspect('equal')

        plt.suptitle(f'{ s.basename }' + 
                     r'  $t$={0:.2f}'.format(d.domain['time']), fontsize='x-large')
        plt.subplots_adjust(top=0.92)

        savdir = osp.join('/tigress/jk11/figures/TIGRESS-FEEDBACK/', s.basename)
        if not osp.exists(savdir):
            os.makedirs(savdir)
        plt.savefig(osp.join(savdir, 'snapshot{0:04d}.png'.format(num)))

        if view:
            break
        else:
            plt.close(plt.gcf())
            
    return ds, savdir

if __name__ == '__main__':
    s = pa.LoadSim('/perseus/scratch/gpfs/jk11/FEEDBACK-TEST/roe.newcool.n50.M1E3')
    ds, savdir = draw_snapshot(s, s.nums[::1], view=False)
