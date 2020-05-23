import os

path = 'cd 12.27'
for x in range(8):
    new_path = ''
    if x == 0:
        new_path = path
    else:
        new_path = path + str(x)
    os.system(new_path)
    os.system('nohup python find_label.py &')
    os.system('cd ..')
