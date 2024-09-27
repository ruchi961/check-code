from dataformer.components.cot import cot
from dataformer.components.rto import rto
from dataformer.components.pvg import pvg
from dataformer.components.SelfConsistency import SelfConsistency
from dataformer.llms import AsyncLLM
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
COLOR = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "PURPLE": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "ENDC": "\033[0m",
}
request_list = [
    {"messages": [{"role": "user", "content": "If a train leaves a station traveling at 60 miles per hour and another train leaves the same station 30 minutes later traveling at 90 miles per hour, when will the second train catch up to the first train?"}]} 
]

llm = AsyncLLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct", api_provider="deepinfra", api_key="T0RWsUfv2lROUOiPi6Vf943Y9RQ4zZ56"
)  

cot = cot(
    llm=llm
)
results = cot.generate()

print("\n\n")
print(f"{COLOR['BLUE']}Prompt: {request_list[0]['messages'][0]['content']}")
print("\n")
for item in results:
    
        print(f"{COLOR['GREEN']}Cot ANswer: {item['cot_response']}{COLOR['ENDC']}")
        print(f"{COLOR['PURPLE']}Model Answer: {item['model_response']}{COLOR['ENDC']}")
        print("\n")


request_list = [
    {"messages": [{"role": "user", "content": "I have a dish of potatoes. The following statements are true: No potatoes of mine, that are new, have >been boiled. All my potatoes in this dish are fit to eat. No unboiled potatoes of mine are fit to eat. Are there any new potatoes in this dish?"}]} 
]


SelfConsistency = SelfConsistency(
    llm=llm
)
results = SelfConsistency.generate()

print("\n\n")
print(f"{COLOR['BLUE']}Prompt: {request_list[0]['messages'][0]['content']}")
print("\n")
for item in results:
    
        print(f"{COLOR['GREEN']}Cot ANswer: {item['SelfConsistency_response']}{COLOR['ENDC']}")
        print(f"{COLOR['PURPLE']}Model Answer: {item['model_response']}{COLOR['ENDC']}")
        print("\n")

request_list = [
    {"messages": [{"role": "user", "content": "Write a code in python for timetable generation. Consider all the constraints."}]} 
]
pvg = pvg(
    llm=llm
)
results = pvg.generate()

print("\n\n")
print(f"{COLOR['BLUE']}Prompt: {request_list[0]['messages'][0]['content']}")
print("\n")
for item in results:
    
        print(f"{COLOR['GREEN']}Cot ANswer: {item['pvg_response']}{COLOR['ENDC']}")
        print(f"{COLOR['PURPLE']}Model Answer: {item['model_response']}{COLOR['ENDC']}")
        print("\n")
request_list = [
    {"messages": [{"role": "user", "content": "Write a genetic algorithm code in python which is fast."}]} 
]

rto = rto(
    llm=llm
)
results = rto.generate()

print("\n\n")
print(f"{COLOR['BLUE']}Prompt: {request_list[0]['messages'][0]['content']}")
print("\n")
for item in results:
    
        print(f"{COLOR['GREEN']}Cot ANswer: {item['rto_response']}{COLOR['ENDC']}")
        print(f"{COLOR['PURPLE']}Model Answer: {item['model_response']}{COLOR['ENDC']}")
        print("\n")