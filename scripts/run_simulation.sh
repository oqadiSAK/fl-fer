# Start the flower server
echo "Starting flower server in background..."
flower-server --insecure > /dev/null 2>&1 &
sleep 2

# Number of client processes to start
N=2
echo "Starting $N clients in background..."

# Start N client processes
for i in $(seq 1 $N)
do
  python scripts/simulation_client.py > /dev/null 2>&1 &
  sleep 1
done

sleep 2
echo "Starting driver..."
python scripts/simulation_driver.py

echo "Clearing background processes..."

# Kill any currently running client
pkill -f 'python scripts/simulation_client.py'

# Kill any currently running flower-server
pkill -f 'flower-server --insecure'