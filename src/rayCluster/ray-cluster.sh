#!/bin/bash

#------------------------------------------------------------------------
#------------------------- Ray Cluster Setup ----------------------------
#------------------------------------------------------------------------

ray stop --force
killall -9 pbs_tmrsh > /dev/null 2>&1

cd $PBS_O_WORKDIR

scriptPath=$(dirname "$0")
UHOME=`eval echo "~$USER"`
nodeDnsIps=`cat $PBS_NODEFILE | uniq`
headNodeDnsIp=`echo $nodeDnsIps | awk '{print $1}'`
headNodeIp=`hostname -i`
rayDashboardPort=7711
rayPort=6379
headNodeIpNport="${headNodeIp}:${rayPort}"
redisPassword=$(uuidgen)

cat > $PBS_O_WORKDIR/setupRayWorkerNode$PBS_ARRAY_INDEX.sh << 'EOF'
#!/bin/bash -l
set -e
ulimit -s unlimited
cd $PBS_O_WORKDIR

headNodeIpNport=${1}
redisPassword=${2}
UHOME=${3}
scriptPath=${4}
ncpus=`$scriptPath/ncpus.sh physical`

source $UHOME/.bashrc
module restore MESA > /dev/null 2>&1
export PATH=$PATH:$UHOME/.local/bin

thisNodeIp=`hostname -i`
ray start --address=$headNodeIpNport --num-cpus=$ncpus \
--redis-password=$redisPassword --block >> ray.log 2>&1
EOF

chmod +x $PBS_O_WORKDIR/setupRayWorkerNode$PBS_ARRAY_INDEX.sh

echo "Setting up Ray cluster......"
J=0
for nodeDnsIp in `echo $nodeDnsIps`
do
        if [[ $nodeDnsIp == $headNodeDnsIp ]]
        then
                echo -e "\nStarting ray cluster on head node..."
                source $UHOME/.bashrc
                ncpus=`$scriptPath/ncpus.sh physical`
                ray start --head --num-cpus=$ncpus --port=$rayPort --redis-password=$redisPassword --include-dashboard=true \
                --dashboard-host=0.0.0.0 --dashboard-port=${rayDashboardPort} 2>&1 | tee >(sed -r 's/\x1b\[[0-9;]*m//g' > ray.log)
                sleep 3
        else
                echo -e "\nStarting ray cluster on worker node $J: $nodeDnsIp" | tee -a ray.log
                pbs_tmrsh $nodeDnsIp $PBS_O_WORKDIR/setupRayWorkerNode$PBS_ARRAY_INDEX.sh $headNodeIpNport $redisPassword $UHOME $scriptPath &
                echo Done. | tee -a ray.log
        fi
        J=$((J+1))
done


echo -e "\nRay cluster setup complete.\n\
Forward the ray dashboard port to localhost using the following command:\n\
ssh -N -L 7711:0.0.0.0:$rayDashboardPort $USER@$headNodeDnsIp -J $USER@_gateway_\n\
Then open the following link in your browser:\n\
http://localhost:8080\n" | tee -a ray.log

export RAY_ADDRESS=$headNodeIpNport

sleep 10
rm $PBS_O_WORKDIR/setupRayWorkerNode$PBS_ARRAY_INDEX.sh