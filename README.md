
# Deep Reinforcment Learning based offloading and allocation

This project simulates the execution of workflows on IoT, fog, and server devices, taking into account factors such as execution delay, communication delay, and cost. The simulation outputs detailed visualizations of the workflows, showing which device executed each task and the associated delays and costs.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Simulation Details](#simulation-details)
- [Visualization](#visualization)
- [Example Workflows](#example-workflows)
- [License](#license)

## Introduction

The goal of this project is to simulate the execution of tasks in workflows distributed across IoT devices, fog devices, and server devices. Each device type has different processing capabilities (measured in MIPS), costs, and communication delays. The simulation tracks the total execution delay and cost for each workflow and provides a detailed visualization of the execution process.

## Features

- Simulation of task execution on IoT, fog, and server devices.
- Calculation of total execution delay and cost.
- Visualization of workflows showing task execution details.
- Adjustable parameters for device capabilities and costs.
- Support for multiple workflows with complex task dependencies.

## Installation

1. **Clone the repository:**
    ``` sh
    git clone https://github.com/your-username/iot-fog-server-simulation.git
    cd iot-fog-server-simulation

2. **Install the required packages:**
    ``` sh
    pip install requirements.txt
    

3. **Ensure Graphviz is installed on your system:**
    - Windows: Download and install Graphviz from [Graphviz.org](https://graphviz.org/download/).
    - macOS: Install via Homebrew:
        \`\`\`sh
        brew install graphviz
        \`\`\`
    - Linux: Install via your package manager (e.g., \`apt\`, \`yum\`).

## Usage

To run the simulation, simply execute the provided Python script:

```sh
python main.py

The script will:
- Simulate the execution of three example workflows.
- Output the total delay and cost for each workflow.
- Generate visualizations of the workflows.
````

## Simulation Details

- **Devices:**
    - **IoT Devices:** 500 MIPS, no cost.
    - **Fog Devices:** 4000 MIPS, $1 per hour.
    - **Server Devices:** 6000 MIPS, $8 per hour.
- **Delays:**
    - **IoT to Broker:** 10 ms
    - **Broker to Fog:** 100 ms
    - **Broker to Server:** 300 ms
- **Workflows:**
    - Each workflow is a directed acyclic graph (DAG) of tasks.
    - Tasks are randomly assigned to IoT devices and may be offloaded to fog or server devices based on random decisions.
    - Total delay and cost are calculated based on device capabilities and communication delays.

## Visualization

The simulation generates visualizations of the workflows using Graphviz. Each task node in the visualization includes:
- Task ID
- Executing device
- Execution cost
- Execution delay
- Communication delay

The visualizations are saved as \`.gv\` files and can be viewed using Graphviz tools.


## Example Workflows

The simulation includes three example workflows:

1. **Workflow 1:**
    - Task 1: 200M instructions
    - Task 2: 590M instructions
    - Task 3: 300M instructions
    - Task 4: 400M instructions
    - Task 5: 250M instructions

2. **Workflow 2:**
    - Task 1: 350M instructions
    - Task 2: 450M instructions
    - Task 3: 600M instructions
    - Task 4: 500M instructions
    - Task 5: 700M instructions
    - Task 6: 550M instructions

3. **Workflow 3:**
    - Task 1: 150M instructions
    - Task 2: 250M instructions
    - Task 3: 400M instructions
    - Task 4: 450M instructions
    - Task 5: 500M instructions
    - Task 6: 350M instructions
    - Task 7: 600M instructions
    - Task 8: 300M instructions


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
