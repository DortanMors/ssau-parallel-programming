import os
for i in range(12):
	for ranks in ['3', '6']:
		os.system(f'mpirun -n {ranks} -oversubscribe -q ./communismMPI q')
		os.system(f'mpirun -n {ranks} -oversubscribe -q ./pointPointMPI q')

