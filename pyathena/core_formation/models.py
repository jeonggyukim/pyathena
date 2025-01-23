m1 = {f"M5J2P{iseed}N512": f"/scratch/gpfs/sm69/cores/hydro/M5.J2.P{iseed}.N512" for iseed in range(0, 40)}
m2 = {f"M10J4P{iseed}N1024": f"/scratch/gpfs/sm69/cores/hydro/M10.J4.P{iseed}.N1024" for iseed in range(0, 2)}
m3 = {f"M10J4P{iseed}N1024": f"/projects2/EOSTRIKE/sanghyuk/cores/M10.J4.P{iseed}.N1024" for iseed in range(2, 7)}
mach5 = m1
mach10 = {**m2, **m3}
hydro = {**mach5, **mach10}
mhd = {f"M10J4B4P{iseed}N1024": f"/scratch/gpfs/sm69/cores/mhd/M10.J4.B4.P{iseed}.N1024" for iseed in range(0, 2)}
models = {**hydro, **mhd}
