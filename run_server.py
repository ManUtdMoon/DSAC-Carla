import subprocess
import argparse
import shlex

parser = argparse.ArgumentParser(description='Program to Run Carla Servers in Docker')
parser.add_argument('--starting_port', type=int, help='starting port', default='2000')
# parser.add_argument('--ids-gpus', type=str, help='string containing the gpu ids', required=True)
parser.add_argument('--num-servers', type=int, help='number of servers', default=3)
args = parser.parse_args()

for i in range(args.num_servers):
    port = args.starting_port + i*3
    if i % 2 == 0:
        gpu_id = 0
    else:
        gpu_id = 1
    # gpu_id = 0
    cmd = "docker run --rm -d -p {}-{}:{}-{} --gpus \'\"device={}\"\' carlasim/carla:0.9.6 /bin/bash CarlaUE4.sh -world-port={}".format(port, port+2, port, port+2, gpu_id, port)
    # print(shlex.split(cmd))
    subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
