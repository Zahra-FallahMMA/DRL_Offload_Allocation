import random
import numpy as np
from collections import deque, defaultdict
from graphviz import Digraph

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def choose_action(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        q_values = self.q_table[state]
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state):
        q_update = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_update - self.q_table[state][action])
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
    def __init__(self, num_iot, num_fog, num_server):
        self.iot_devices = [Device(f'iot_{i}', 500, 0) for i in range(num_iot)]
        self.fog_devices = [Device(f'fog_{i}', 4000, 1) for i in range(num_fog)]
        self.server_devices = [Device(f'server_{i}', 6000, 8) for i in range(num_server)]
        self.total_delay = 0
        self.total_cost = 0
        self.workflows = []
        self.ready_tasks = defaultdict(deque)
        self.iot_agent = QLearningAgent(state_size=2, action_size=2)  # State: [current_cost, current_delay], Actions: [execute, offload]
        self.broker_agent = QLearningAgent(state_size=2, action_size=2)  # State: [current_cost, current_delay], Actions: [fog, server]

    def add_workflow(self, workflow):
        self.workflows.append(workflow)
        # Assign the workflow to an IoT device randomly
        iot_device = random.choice(self.iot_devices)
        print(f"Workflow {workflow.id} assigned to {iot_device.id}")
        for task_id, task in workflow.tasks.items():
            if not task.parents:
                self.ready_tasks[iot_device.id].append(task)
                print(f"Task {task_id} is ready to be executed on {iot_device.id}")

    def execute_task(self, task, device, comm_delay):
        execution_time = task.instructions / device.mips / 1e6  # Convert MIPS to instructions per second
        task.execution_time = execution_time + comm_delay / 1000  # Adding communication delay in seconds
        task.comm_delay = comm_delay / 1000  # Store communication delay in seconds
        if device.cost_per_hour == 0:
            delay = task.execution_time  # Delay in seconds for IoT
            self.total_delay += delay
            task.cost = 0
            print(f"Task {task.id} executed on IoT device {device.id} with delay {delay:.2f} seconds")
        else:
            cost = execution_time * device.cost_per_hour
            self.total_cost += cost
            task.cost = cost
            self.total_delay += comm_delay / 1000  # Adding communication delay in seconds
            print(f"Task {task.id} offloaded to {device.id} with cost ${cost:.2f} and delay {comm_delay} ms")

        task.executed_on = device.id
        task.executed = True

    def check_ready_tasks(self, task, device_id):
        for child in task.children:
            if all(parent.executed for parent in child.parents):
                self.ready_tasks[device_id].append(child)
                print(f"Task {child.id} is now ready to be executed on {device_id}")

    def simulate(self):
        print("Simulation started.")
        while any(self.ready_tasks.values()):
            for device_id in list(self.ready_tasks.keys()):
                if not self.ready_tasks[device_id]:
                    continue

                task = self.ready_tasks[device_id].popleft()
                print(f"Task {task.id} is being processed on {device_id}")
                # State for the IoT agent: [current total cost, current total delay]
                state = (self.total_cost, self.total_delay)
                # Decide if the task is executed on IoT or offloaded
                action = self.iot_agent.choose_action(state)
                if action == 0:  # Execute on IoT
                    iot_device = next(device for device in self.iot_devices if device.id == device_id)
                    self.execute_task(task, iot_device, comm_delay=0)
                    reward = -task.execution_time  # Reward based on execution time
                    self.iot_agent.learn(state, action, reward, (self.total_cost, self.total_delay))
                    self.check_ready_tasks(task, device_id)
                else:  # Offload
                    # Offload the task to broker
                    broker_delay = 10  # Communication delay between IoT and Broker in ms
                    broker_state = (self.total_cost, self.total_delay)
                    broker_action = self.broker_agent.choose_action(broker_state)
                    if broker_action == 0:  # Execute on Fog
                        fog_device = random.choice(self.fog_devices)
                        self.execute_task(task, fog_device, comm_delay=broker_delay + 100)  # Adding delay between Broker and Fog in ms
                        reward = -task.execution_time - task.cost  # Reward based on execution time and cost
                        self.broker_agent.learn(broker_state, broker_action, reward, (self.total_cost, self.total_delay))
                    else:  # Execute on Server
                        server_device = random.choice(self.server_devices)
                        self.execute_task(task, server_device, comm_delay=broker_delay + 300)  # Adding delay between Broker and Server in ms
                        reward = -task.execution_time - task.cost  # Reward based on execution time and cost
                        self.broker_agent.learn(broker_state, broker_action, reward, (self.total_cost, self.total_delay))
                    self.check_ready_tasks(task, device_id)
                    self.iot_agent.learn(state, action, reward, (self.total_cost, self.total_delay))

        print(f"Total Delay: {self.total_delay:.2f} seconds")
        print(f"Total Cost: ${self.total_cost:.2f}")
        print("Simulation completed.")
        self.visualize_workflows()

    def visualize_workflows(self):
        for workflow in self.workflows:
            dot = Digraph(comment=f'Workflow {workflow.id}')
            for task_id, task in workflow.tasks.items():
                label = (
                    f'{task.id}\n'
                    f'Device: {task.executed_on}\n'
                    f'Cost: ${task.cost:.2f}\n'
                    f'Delay: {task.execution_time:.2f}s\n'
                    f'Comm Delay: {task.comm_delay:.2f}s'
                )
                dot.node(task.id, label)
                for child in task.children:
                    dot.edge(task.id, child.id)
            dot.render(f'workflow_{workflow.id}.gv', view=True)
            print(f"Workflow {workflow.id} visualization saved.")

# Example Usage
simulation = Simulation(num_iot=3, num_fog=2, num_server=2)

# Create a workflow (example)
workflow1 = Workflow(id='workflow_1')
workflow1.add_task(task_id='task_1', instructions=200e6)  # Root task
workflow1.add_task(task_id='task_2', instructions=590e6, parent_ids=['task_1'])
workflow1.add_task(task_id='task_3', instructions=300e6, parent_ids=['task_1'])
workflow1.add_task(task_id='task_4', instructions=400e6, parent_ids=['task_2'])
workflow1.add_task(task_id='task_5', instructions=250e6, parent_ids=['task_2', 'task_3'])

# Create a second workflow
workflow2 = Workflow(id='workflow_2')
workflow2.add_task(task_id='task_1', instructions=350e6)  # Root task
workflow2.add_task(task_id='task_2', instructions=450e6, parent_ids=['task_1'])
workflow2.add_task(task_id='task_3', instructions=600e6, parent_ids=['task_1'])
workflow2.add_task(task_id='task_4', instructions=500e6, parent_ids=['task_2'])
workflow2.add_task(task_id='task_5', instructions=700e6, parent_ids=['task_3'])
workflow2.add_task(task_id='task_6', instructions=550e6, parent_ids=['task_4', 'task_5'])

# Create a third workflow
workflow3 = Workflow(id='workflow_3')
workflow3.add_task(task_id='task_1', instructions=150e6)  # Root task
workflow3.add_task(task_id='task_2', instructions=250e6, parent_ids=['task_1'])
workflow3.add_task(task_id='task_3', instructions=400e6, parent_ids=['task_1'])
workflow3.add_task(task_id='task_4', instructions=450e6, parent_ids=['task_2'])
workflow3.add_task(task_id='task_5', instructions=500e6, parent_ids=['task_3'])
workflow3.add_task(task_id='task_6', instructions=350e6, parent_ids=['task_4', 'task_5'])
workflow3.add_task(task_id='task_7', instructions=600e6, parent_ids=['task_5'])
workflow3.add_task(task_id='task_8', instructions=300e6, parent_ids=['task_6', 'task_7'])

# Add workflows to simulation
simulation.add_workflow(workflow1)
simulation.add_workflow(workflow2)
simulation.add_workflow(workflow3)

# Run the simulation
simulation.simulate()
