import json
import urllib.request

from tqdm import tqdm

def query_model(
        prompt,
        model = 'llama3',
        url = 'http://localhost:11434/api/chat'
):
    data = {
        'model':model,
         'messages':[
             {'role':'user','content':prompt}
         ],
        'options':{
            'seed':123,
            'temperature':0,
            'num_ctx':2048
        }
    }

    payload = json.dumps(data).encode('utf-8')
    request = urllib.request.Request(
        url,
        data = payload,
        method='POST'
    )
    request.add_header('Content-Type' , 'application/json')

    response_data = ''

    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode('utf-8')
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json['message']['content']
    return response_data


file_path = "instruction-data.json"
with open(file_path, "r") as file:
    test_data = json.load(file)

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )


    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

model = "llama3"
result = query_model("What do Llamas eat?", model)
print(result)
