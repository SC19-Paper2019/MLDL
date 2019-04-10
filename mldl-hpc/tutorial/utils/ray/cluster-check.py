import time
import ray
import sys

head = sys.argv[1] 
redis_address = "%s:6379"%head
ray.init(redis_address=redis_address)
@ray.remote
def f():
    time.sleep(0.01)
    return ray.services.get_node_ip_address()

# Get a list of the IP addresses of the nodes that have joined the cluster.
nodes=set(ray.get([f.remote() for _ in range(1000)]))
print(nodes)
