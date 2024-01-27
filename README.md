# A Federated Learning Platform For Facial Expression Recognition

This project aims to create a federated learning platform for facial expression recognition. The platform is built using the [Flower framework](https://flower.dev/) and allows users to contribute to the local dataset and improve the model without sharing their actual data. It consists of three main components:

1. **Centralized Approach**

2. **Simulation**

3. **Platform**

## Prerequisites

Before you begin, ensure you have met the following requirements:

1. **Download the FER2013 dataset**: This project uses the FER2013 dataset. You can download it from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). After downloading, place the `fer2013.csv` file in the `/data` folder of this project.

2. **Prepare the dataset**: After placing the dataset in the correct folder, you need to split it into train, test, and validation sets. This can be done by running the `split.py` script. Open your terminal and execute the following command:

    ```sh
    python scripts/split.py
    ```

3. **Install the necessary dependencies**: This project requires certain Python packages to run. You can install these by running the following command in your terminal:

    ```sh
    pip install -r requirements.txt
    ```

## Centralized Approach

This project includes a simple neural network as a baseline model. The model is designed to be lightweight and efficient, making it suitable for deployment on edge devices like the Raspberry Pi. It is trained using the FER2013 dataset.

### Running the Centralized Training

You can initiate the centralized training process by running the following command in your terminal:

```sh
python scripts/centralized.py
```

### Training Parameters

The training script accepts several optional parameters. If not provided, default values will be used. Here's a description of each parameter:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--epochs` | int | 200 | The number of complete passes through the training dataset |
| `--learning_rate` | float | 0.01 | The speed at which the model learns. A lower value means slower learning, but too high a value may result in unpredictable behavior |
| `--momentum` | float | 0.9 | The factor used to reduce oscillations in the gradient descent |
| `--weight_decay` | float | 5e-3 | The regularization term. It helps prevent the model from overfitting |

## Simulation
The simulation is a crucial part of this project as it allows us to test and validate the federated learning process before its integration into a continuous, asynchronous platform. It uses the Flower framework to simulate a federated learning environment with multiple clients learning from the FER2013 dataset.

In the simulation, each client is initialized with the same model and then trained on the FER2013 dataset. After each round of training, the clients send their updated models back to the server. The server then aggregates these models into a single global model using a process(`strategy`) called Federated Averaging. This global model is then sent back to the clients for the next round of training. This process is repeated for a `predefined number of rounds` in a `synchronized` flow.

You can run the simulation by executing the shell script:

```sh
./scripts/run_simulation.sh
```

The number of clients participating in the simulation can be adjusted within the script.


## Platform

The platform is the main component of this project. It is designed to facilitate federated learning for facial expression recognition on edge devices, such as Raspberry Pi.

The platform consists of three core components: the server, the client, and the driver.

- The **server** is responsible for coordinating the federated learning process. It receives updated models from the clients, aggregates them into a global model, and sends this global model back to the clients.

- The **client** is a GUI that allows users to contribute to the local dataset and participate in the federated learning process. It captures video, detects faces, and identifies emotions using the global model received from the server.

- The **driver** orchestrates the federated learning process. It triggers the start of each training round and manages the communication between the server and the clients.

The platform operates asynchronously, which means that when a federated learning round is triggered, not all clients are required to participate in local training. This is because some clients may not have local data to contribute to the learning process.

<p align="center">
  <img src="https://github.com/oqadiSAK/fl-fer/assets/107847428/2182444b-35a1-4242-8d68-e8121f6eceed" alt="FL Platform Workflow"/>
</p>

### Server

The server uses Flower's server (`flower-server`), which operates continuously and awaits orchestration by a driver script. More details about this server can be found in Flower's [documentation](https://flower.dev/docs/framework/ref-api-cli.html#flower-server). For this project, we'll use the straightforward non-secure mode. You can start the server with the following command:

```sh
flower-server --insecure
```

### Client

The client is a user-friendly GUI developed using Qt. It displays video, detects faces using OpenCV, and identifies emotions using the model. The client offers features such as saving the current frame to local training data, with the option to label it with a selected emotion. Additionally, users have the capability to initiate federated learning.

The client program requires the `--cam-type` parameter, which can take two values: `pi` or `cv`. Use `pi` for the Raspberry Pi camera module (this uses the Picamera2 library), and `cv` for general purposes (this uses OpenCV). For example, you can run the client with the following command:

```sh
python app.py --cam-type=cv
```

Here are the parameters you can use when running the client:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--cam-type` | string | N/A | Camera type to use: 'pi' for Picamera2, 'cv' for OpenCV VideoCapture (required) |
| `--server_ip` | string | 'localhost' | Server address |
| `--server_port` | int | 9092 | Server port |
| `--driver_ip` | string | 'localhost' | Driver address |
| `--driver_port` | int | 9093 | Driver port |
| `--threshold` | int | 5 | Threshold value representing the minimum number of images to participate in the federated learning |

All parameters except `--cam-type` are optional. If not provided, the default values will be used.

### Driver

The driver can be considered as a server implementation for a custom federated learning workflow. It orchestrates the federated learning process. The driver waits for clients to connect and then waits for the command to trigger federated learning. 

The need for a driver arises from the limitations of the Flower framework used for building Federated Learning. Currently, Flower lacks native support for asynchronous federated learning, hence the need for a custom solution.

For example, you can run the driver with the following command:

```sh
python run_driver.py --port=9093 --server_ip=localhost --server_port=9091
```
Here are the parameters you can use when running the driver:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--port` | int | 9093 | Port |
| `--server_ip` | string | 'localhost' | Server IP |
| `--server_port` | int | 9091 | Server port |

All parameters are optional. If not provided, the default values will be used.

## Running the Platform

To successfully run the platform, you need to start the server, driver, and at least two clients in the correct order. Please note that each client requires its own device with a camera.

Here you can see the platform application in action:
<p align="center">
  <img src="https://github.com/oqadiSAK/fl-fer/assets/107847428/55e27c08-bfe1-42e1-94cf-cd297380b011" alt="FL Platform"/>
</p>
