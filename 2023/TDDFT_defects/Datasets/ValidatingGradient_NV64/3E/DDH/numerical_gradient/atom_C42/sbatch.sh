#!/bin/bash
#SBATCH --job-name=ddh-42
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=24:00:00
#SBATCH --partition=gagalli-csl
##SBATCH --qos=gagalli-debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40

module load intel/19.1.1
module load intelmpi/2019.up7+intel-19.1.1
module load mkl/2020.up1
module load python/cpython-3.8.5

#export I_MPI_PMI_LIBRARY=/software/slurm-current-$DISTARCH/lib/libpmi.so
export LD_LIBRARY_PATH=/software/python-3.8.5-el7-x86_64/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1

QEDIR=/project2/gagalli/jinyu/WEST-Develop/WEST-5.4.0-TDDFT/QEdir7/bin

for i in xm1  xm2  xp1  xp2  ym1  ym2  yp1  yp2  zm1  zm2  zp1  zp2
do
cd $i
mpirun -n 80 ${QEDIR}/pw.x -nb 2 < pw.in > pw.out
mpirun -n 80 ${QEDIR}/wbse_init.x -ni 2 -i wbse_init.in > wbse_init.out
mpirun -n 80 ${QEDIR}/wbse.x -nb 2 -i wbse.in > wbse.out
cd ..
done
