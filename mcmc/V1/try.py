

with open('state_params.txt', 'r') as file:
    lines = file.readlines()

params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]
print(params)