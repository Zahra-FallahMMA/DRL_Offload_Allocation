import xml.etree.ElementTree as ET
from io import StringIO

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

def parse_dax(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    workflow_id = root.attrib.get('name')
    workflow = Workflow(workflow_id)

    # Parse jobs
    jobs = {job.attrib['id']: job for job in root.findall('{http://pegasus.isi.edu/schema/DAX}job')}

    # Add jobs to workflow
    for job_id, job in jobs.items():
        instructions = float(job.attrib.get('runtime', 0))
        workflow.add_task(job_id, instructions)

    # Parse dependencies
    for child in root.findall('{http://pegasus.isi.edu/schema/DAX}child'):
        child_id = child.attrib['ref']
        parent_ids = [parent.attrib['ref'] for parent in child.findall('{http://pegasus.isi.edu/schema/DAX}parent')]
        # Add task with its parents
        workflow.add_task(child_id, 0, parent_ids)  # Added 0 as instructions for child node because it shouldn't overwrite

    return workflow

def print_dependencies(workflow):
    for task_id, task in workflow.tasks.items():
        parent_ids = [parent.id for parent in task.parents]
        child_ids = [child.id for child in task.children]
        print(f"Task {task_id}:")
        print(f"  Parents: {parent_ids if parent_ids else 'None'}")
        print(f"  Children: {child_ids if child_ids else 'None'}")
        print("")

'''
# Test Code Begins Here
def test_parse_dax_and_print_dependencies():
    # Creating a mock DAX XML content
    dax_content = """<?xml version="1.0" encoding="UTF-8"?>
    <adag xmlns="http://pegasus.isi.edu/schema/DAX" name="test_workflow" index="0" count="1">
        <job id="task_1" namespace="example" name="task1" version="1.0" runtime="100.0"/>
        <job id="task_2" namespace="example" name="task2" version="1.0" runtime="200.0"/>
        <job id="task_3" namespace="example" name="task3" version="1.0" runtime="300.0"/>
        <child ref="task_2">
            <parent ref="task_1"/>
        </child>
        <child ref="task_3">
            <parent ref="task_2"/>
        </child>
    </adag>"""

    # Using StringIO to simulate reading from file without actual file IO
    dax_file_path = StringIO(dax_content)

    # Parse the DAX content
    workflow = parse_dax(dax_file_path)

    # Print dependencies
    print_dependencies(workflow)

# Running the test
test_parse_dax_and_print_dependencies()
'''