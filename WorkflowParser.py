import xml.etree.ElementTree as ET

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
        for parent_id in parent_ids:
            workflow.add_task(child_id, {}, parent_ids)

    return workflow