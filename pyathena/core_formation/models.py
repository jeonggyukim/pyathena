m1 = {f"M5J2P{iseed}N512": f"/scratch/gpfs/sm69/cores/hydro/M5.J2.P{iseed}.N512" for iseed in range(0, 40)}
m2 = {f"M10J4P{iseed}N1024": f"/scratch/gpfs/sm69/cores/hydro/M10.J4.P{iseed}.N1024" for iseed in range(0, 3)}
m3 = {f"M10J4P{iseed}N1024": f"/projects2/EOSTRIKE/coreform_MO2024/M10.J4.P{iseed}.N1024" for iseed in range(3, 7)}
mach5 = m1
mach10 = {**m2, **m3}
models = {**mach5, **mach10}
