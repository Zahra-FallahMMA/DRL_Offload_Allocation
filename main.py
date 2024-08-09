import random
import numpy as np
from collections import deque, defaultdict
from itertools import product
from DQNAgent import DQNAgent
from WorkflowParser import parse_dax

class Device:
    def __init__(self, id, mips, cost_per_hour):
        self.id = id
        self.mips = mips
        self.cost_per_hour = cost_per_hour

class Simulation:
    def __init__(self, num_iot, num_fog, num_server, learning_rate=0.001, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01, batch_size=64, memory_size=2000):
        self.num_iot = num_iot
        self.num_fog = num_fog
        self.num_server = num_server
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.reset()

    def reset(self):
        self.iot_devices = [Device(f'iot_{i}', 500, 0) for i in range(self.num_iot)]
        self.fog_devices = [Device(f'fog_{i}', 4000, 1) for i in range(self.num_fog)]
        self.server_devices = [Device(f'server_{i}', 6000, 8) for i in range(self.num_server)]
        self.total_delay = 0
        self.total_cost = 0
        self.workflows = []
        self.ready_tasks = defaultdict(deque)
        self.iot_agent = DQNAgent(state_size=3, action_size=2, learning_rate=self.learning_rate, discount_factor=self.discount_factor,
                                  exploration_rate=self.exploration_rate, exploration_decay=self.exploration_decay,
                                  exploration_min=self.exploration_min, batch_size=self.batch_size, memory_size=self.memory_size)
        self.broker_agent = DQNAgent(state_size=4, action_size=2, learning_rate=self.learning_rate, discount_factor=self.discount_factor,
                                     exploration_rate=self.exploration_rate, exploration_decay=self.exploration_decay,
                                     exploration_min=self.exploration_min, batch_size=self.batch_size, memory_size=self.memory_size)

    def add_workflow(self, workflow):
        self.workflows.append(workflow)
        # Assign the workflow to an IoT device randomly
        iot_device = random.choice(self.iot_devices)
        for task_id, task in workflow.tasks.items():
            if not task.parents:
                self.ready_tasks[iot_device.id].append(task)

    def execute_task(self, task, device, comm_delay):
        execution_time = task.instructions / device.mips / 1e6  # Convert MIPS to instructions per second
        task.execution_time = execution_time + comm_delay / 1000  # Adding communication delay in seconds
        task.comm_delay = comm_delay / 1000  # Store communication delay in seconds
        if device.cost_per_hour == 0:
            delay = task.execution_time  # Delay in seconds for IoT
            self.total_delay += delay
            task.cost = 0
        else:
            cost = execution_time * device.cost_per_hour
            self.total_cost += cost
            task.cost = cost
            self.total_delay += comm_delay / 1000  # Adding communication delay in seconds
        task.executed_on = device.id
        task.executed = True

    def check_ready_tasks(self, task, device_id):
        for child in task.children:
            if all(parent.executed for parent in child.parents):
                self.ready_tasks[device_id].append(child)

    def simulate(self):
        while any(self.ready_tasks.values()):
            for device_id in list(self.ready_tasks.keys()):
                if not self.ready_tasks[device_id]:
                    continue

                task = self.ready_tasks[device_id].popleft()
                pending_tasks = len(self.ready_tasks[device_id])
                # State for the IoT agent: [current total cost, current total delay, pending tasks]
                state = np.array([self.total_cost, self.total_delay, pending_tasks]).reshape(1, -1)
                # Decide if the task is executed on IoT or offloaded
                action = self.iot_agent.choose_action(state)
                done = False
                if action == 0:  # Execute on IoT
                    iot_device = next(device for device in self.iot_devices if device.id == device_id)
                    self.execute_task(task, iot_device, comm_delay=0)
                    reward = -task.execution_time  # Reward based on execution time
                    next_state = np.array([self.total_cost, self.total_delay, len(self.ready_tasks[device_id])]).reshape(1, -1)
                    if not self.ready_tasks[device_id]:
                        done = True
                    self.iot_agent.remember(state, action, reward, next_state, done)
                    self.iot_agent.replay()
                    self.check_ready_tasks(task, device_id)
                else:  # Offload
                    # Offload the task to broker
                    broker_delay = 10  # Communication delay between IoT and Broker in ms
                    broker_state = np.array([self.total_cost, self.total_delay, broker_delay / 1000, pending_tasks]).reshape(1, -1)
                    broker_action = self.broker_agent.choose_action(broker_state)
                    if broker_action == 0:  # Execute on Fog
                        fog_device = random.choice(self.fog_devices)
                        self.execute_task(task, fog_device, comm_delay=broker_delay + 100)  # Adding delay between Broker and Fog in ms
                        reward = -task.execution_time - task.cost  # Reward based on execution time and cost
                        next_broker_state = np.array([self.total_cost, self.total_delay, broker_delay / 1000, len(self.ready_tasks[device_id])]).reshape(1, -1)
                        if not self.ready_tasks[device_id]:
                            done = True
                        self.broker_agent.remember(broker_state, broker_action, reward, next_broker_state, done)
                        self.broker_agent.replay()
                    else:  # Execute on Server
                        server_device = random.choice(self.server_devices)
                        self.execute_task(task, server_device, comm_delay=broker_delay + 300)  # Adding delay between Broker and Server in ms
                        reward = -task.execution_time - task.cost  # Reward based on execution time and cost
                        next_broker_state = np.array([self.total_cost, self.total_delay, broker_delay / 1000, len(self.ready_tasks[device_id])]).reshape(1, -1)
                        if not self.ready_tasks[device_id]:
                            done = True
                        self.broker_agent.remember(broker_state, broker_action, reward, next_broker_state, done)
                        self.broker_agent.replay()
                    self.check_ready_tasks(task, device_id)
                    next_state = np.array([self.total_cost, self.total_delay, len(self.ready_tasks[device_id])]).reshape(1, -1)
                    if not self.ready_tasks[device_id]:
                        done = True
                    self.iot_agent.remember(state, action, reward, next_state, done)
                    self.iot_agent.replay()

    def run_simulation(self, num_runs=100, dax_path='CyberShake_30.xml'):
        total_delays = []
        total_costs = []

        # Parse workflows from DAX file
        workflow = parse_dax(dax_path)

        for _ in range(num_runs):
            self.reset()

            # Add parsed workflow to simulation
            self.add_workflow(workflow)

            # Run simulation
            self.simulate()

            # Collect results
            total_delays.append(self.total_delay)
            total_costs.append(self.total_cost)

        mean_delay = np.mean(total_delays)
        mean_cost = np.mean(total_costs)
        return mean_delay, mean_cost

def hyperparameter_tuning(num_runs=100):
    learning_rates = [0.001,0.0001]
    discount_factors = [ 0.95]
    exploration_rates = [0.5]
    exploration_decays = [0.99]
    exploration_mins = [ 0.05]

    best_mean_delay = float('inf')
    best_mean_cost = float('inf')
    best_params = None

    for lr, df, er, ed, em in product(learning_rates, discount_factors, exploration_rates, exploration_decays, exploration_mins):
        simulation = Simulation(num_iot=3, num_fog=2, num_server=2, learning_rate=lr, discount_factor=df, exploration_rate=er, exploration_decay=ed, exploration_min=em)
        mean_delay, mean_cost = simulation.run_simulation(num_runs=num_runs)
        print(f"Params: LR={lr}, DF={df}, ER={er}, ED={ed}, EM={em} -> Mean Delay: {mean_delay:.2f}, Mean Cost: ${mean_cost:.2f}")

        if mean_cost < best_mean_cost or (mean_cost == best_mean_cost and mean_delay < best_mean_delay):
            best_mean_delay = mean_delay
            best_mean_cost = mean_cost
            best_params = (lr, df, er, ed, em)

    print(f"Best Params: LR={best_params[0]}, DF={best_params[1]}, ER={best_params[2]}, ED={best_params[3]}, EM={best_params[4]} -> Mean Delay: {best_mean_delay:.2f}, Mean Cost: ${best_mean_cost:.2f}")

# Run hyperparameter tuning
hyperparameter_tuning(num_runs=100)
