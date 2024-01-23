import tqdm
import pandas as pd
import timeit
from datasets.small_arc_dataset import SmallArcDirectGridDataset
import yaml
import os
from pipelines.arcathon_pipeline import ArcathonPipeline
from config import config
from pipelines.build import build_pipeline
from prompting_classes import prompt_cls


def save_to_csv(csv_path, data, success, complete_out):
    experiment_name = csv_path.split('/')[-1].split('.')[0]
    data = data.copy()
    data[f'{experiment_name} success'] = success
    data[f'{experiment_name} complete output'] = complete_out

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(columns=data.index)
    # Append row to the dataframe
    df = pd.concat([df, pd.DataFrame([data])])
    df.to_csv(csv_path)

SAVE_EVERY = 1
def create_directory_for_file(file_path):
    # Extract the directory path from the file path
    directory = os.path.dirname(file_path)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

class ResultsSaver:
    def __init__(self,path):
        self.path = path
        create_directory_for_file(self.path)
        self.results = []
    def append(self,result):
        self.results.append(result)
        if len(self.results) % SAVE_EVERY == 0:
            self.save()

    def save(self):
        before_df =[]
        for result in self.results:
            data,success,saved_outputs = result
            task_id = data['Task_ID']
            d = {'task_id' : task_id,'success' : success,'output' : saved_outputs}
            before_df.append(d)
        df = pd.DataFrame(before_df)
        df.to_csv(self.path)

if __name__ == '__main__':
    dataset = SmallArcDirectGridDataset(config['dataset_location'])
    # print('Running with the following config:')
    # print(yaml.dump(config, default_flow_style=False))
    prompt_converter = prompt_cls(**config.prompt_config)
    pipeline = build_pipeline(config.arcathon_pipeline.name, prompt_convertor=prompt_converter,
                              **config.arcathon_pipeline.args)
    output_path = os.path.join(config['output_path'],str(prompt_converter),str(pipeline))+'.csv'
    if os.path.exists(output_path):
        print(f'Outputh path {output_path} already exists. Exiting...')
        exit()
    saver = ResultsSaver(output_path)

    print('Output path is:',output_path)
    print('Starting...')
    success_amount = 0

    for idx in tqdm.tqdm(range(len(dataset))): #data, trains, tests in tqdm.tqdm(dataset):
        data, trains, tests = dataset[idx]

        print('Starting to work on', data['Task_ID'])
        success, saved_outputs = pipeline(trains, tests)
        if success:
            success_amount += 1
        print(
            f'Finished working on {data["Task_ID"]}, success: {success}, in index {idx}/{len(dataset)} and so far {success_amount} successes.')
        saver.append((data, success, saved_outputs))
        idx += 1
