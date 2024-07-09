import  pyautogui as auto
import time
import json
import pywhatkit
import os
def execute_command(command):
    if command == "none": return
    if command == None: return
    split1 = command.split(";")
    commands = list()
    for cm in split1:
        commands.append(cm.split(','))

    for cm in commands:
        if cm[0] == "move":
            auto.mouseUp(int(cm[1]), int(cm[2]))
        if cm[0] == "click":
            auto.leftClick()
        if cm[0] == "wait":
            time.sleep(float(cm[1]))
        if cm[0] == "location":
            print(auto.position())
        if cm[0] == "type":
            auto.write(cm[1])
        if cm[0] == 'press':
            auto.press(cm[1])
        if cm[0] == "send":
            pywhatkit.sendwhatmsg_instantly(cm[1], cm[2],tab_close=True, wait_time=15)
        if cm[0] == "run":
            os.system(cm[1])

        time.sleep(0.5)

def id_to_command(id):
    f = open(r"C:\Users\orile\Repos\hand-recognition\command.json",'r')
    commands = json.load(f)
    for i in commands["commands"]:
        if i["id"] == id:
            print(i["command"])
            execute_command(i["command"])
    

