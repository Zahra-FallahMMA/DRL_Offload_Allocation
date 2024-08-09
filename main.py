import random
import numpy as np
from collections import deque, defaultdict
from graphviz import Digraph
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from itertools import product

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01, batch_size=64, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)
                target[0][action] = reward + self.discount_factor * np.amax(t[0])
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class Task:
    def __init__(self, id, instructions):
        self.id = id
        self.instructions = instructions
        self.children = []
        self.parents = []
        self.executed = False
        self.executed_on = None
        self.execution_time = 0
        self.cost = 0
        self.comm_delay = 0  # Communication delay in seconds

class Workflow:
    def __init__(self, id):
        self.id = id
        self.tasks = {}

    def add_task(self, task_id, instructions, parent_ids=[]):
        if task_id not in self.tasks:
            self.tasks[task_id] = Task(task_id, instructions)
        task = self.tasks[task_id]
        for parent_id in parent_ids:
            if parent_id not in self.tasks:
                self.tasks[parent_id] = Task(parent_id, 0)
            parent_task = self.tasks[parent_id]
            parent_task.children.append(task)
            task.parents.append(parent_task)

class Device:
    def __init__(self, id, mips, cost_per_hour):
        self.id = id
        self.mips = mips
        self.cost_per_hour = cost_per_hour

class Simulation:
    def __init__(self, num_iot, num_fog, num_server, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01, batch_size=64, memory_size=2000):
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
        self.iot_agent = DQNAgent(state_size=3, action_size=2, learning_rate=self.learning_rate, discount_factor=self.discount_factor, exploration_rate=self.exploration_rate, exploration_decay=self.exploration_decay, exploration_min=self.exploration_min, batch_size=self.batch_size, memory_size=self.memory_size)
        self.broker_agent = DQNAgent(state_size=4, action_size=2, learning_rate=self.learning_rate, discount_factor=self.discount_factor, exploration_rate=self.exploration_rate, exploration_decay=self.exploration_decay, exploration_min=self.exploration_min, batch_size=self.batch_size, memory_size=self.memory_size)

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

    def run_simulation(self, num_runs=100):
        total_delays = []
        total_costs = []
        for _ in range(num_runs):
            self.reset()

            # Create workflows
            workflow1 = Workflow(id='workflow_1')
            workflow1.add_task(task_id='task_1', instructions=200e6)  # Root task
            workflow1.add_task(task_id='task_2', instructions=590e6, parent_ids=['task_1'])
            workflow1.add_task(task_id='task_3', instructions=300e6, parent_ids=['task_1'])
            workflow1.add_task(task_id='task_4', instructions=400e6, parent_ids=['task_2'])
            workflow1.add_task(task_id='task_5', instructions=250e6, parent_ids=['task_2', 'task_3'])

            workflow2 = Workflow(id='workflow_2')
            workflow2.add_task(task_id='task_1', instructions=350e6)  # Root task
            workflow2.add_task(task_id='task_2', instructions=450e6, parent_ids=['task_1'])
            workflow2.add_task(task_id='task_3', instructions=600e6, parent_ids=['task_1'])
            workflow2.add_task(task_id='task_4', instructions=500e6, parent_ids=['task_2'])
            workflow2.add_task(task_id='task_5', instructions=700e6, parent_ids=['task_3'])
            workflow2.add_task(task_id='task_6', instructions=550e6, parent_ids=['task_4', 'task_5'])

            workflow3 = Workflow(id='workflow_3')
            workflow3.add_task(task_id='task_1', instructions=150e6)  # Root task
            workflow3.add_task(task_id='task_2', instructions=250e6, parent_ids=['task_1'])
            workflow3.add_task(task_id='task_3', instructions=400e6, parent_ids=['task_1'])
            workflow3.add_task(task_id='task_4', instructions=450e6, parent_ids=['task_2'])
            workflow3.add_task(task_id='task_5', instructions=500e6, parent_ids=['task_3'])
            workflow3.add_task(task_id='task_6', instructions=350e6, parent_ids=['task_4', 'task_5'])
            workflow3.add_task(task_id='task_7', instructions=600e6, parent_ids=['task_5'])
            workflow3.add_task(task_id='task_8', instructions=300e6, parent_ids=['task_6', 'task_7'])

            self.add_workflow(workflow1)
            self.add_workflow(workflow2)
            self.add_workflow(workflow3)

            # Run simulation
            self.simulate()

            # Collect results
            total_delays.append(self.total_delay)
            total_costs.append(self.total_cost)

        mean_delay = np.mean(total_delays)
        mean_cost = np.mean(total_costs)
        return mean_delay, mean_cost

def hyperparameter_tuning(num_runs=100):
    learning_rates = [0.001, 0.01, 0.1]
    discount_factors = [0.8, 0.9, 0.95]
    exploration_rates = [1.0, 0.5]
    exploration_decays = [0.99, 0.995]
    exploration_mins = [0.01, 0.05]

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
