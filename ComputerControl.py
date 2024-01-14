import  pyautogui as auto
import time
import json

def execute_command(command):
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
        time.sleep(0.5)

def id_to_command(id):
    f = open(r"C:\Users\orile\hand-recognition\command.json",)
    commands = json.load(f)
    for i in commands["commands"]:
        if i["id"] == id:
            return i["command"]

