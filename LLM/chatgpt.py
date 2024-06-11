from openai import OpenAI
import re
import math, copy
import numpy as np
import os, sys, json, time, shutil
import rclpy
sys.path.append("./")
from Utils.wait_ros2 import wait_for_message
from std_msgs.msg import String
from collections import OrderedDict

class GPT():
    def __init__(self):

        self.node = rclpy.create_node("gpt_controller")
        self.logger = self.node.get_logger()
        _,robot_name=wait_for_message(String, self.node, "/start_llm")
        self.logger.info("got robot name")
        _,abstraction=wait_for_message(String, self.node, "/abstraction")
        self.logger.info("got abstraction level")
        self.responsePublisher=self.node.create_publisher(String,"/llm_response",10)
        self.execPublisher=self.node.create_publisher(String, "/run_gpt_code", 1)

        self.robot_name=robot_name.data.lower()
        self.abstraction=abstraction.data.lower()

        self.reset_count=0
        time.sleep(3)

        open('./LLM/chat_history.txt', 'w').close()

        with open("./LLM/Lib/"+self.robot_name+"/config.json", "r") as f:
            self.config = json.load(f)

        self.logger.info("Initializing ChatGPT...")

        self.client = OpenAI(api_key=self.config["OPENAI_API_KEY"])
        self.last_response=''

        self.sysprompt_path = "./LLM/Lib/"+self.robot_name+"/system_prompts/basic.txt"
        self.prompt_path = "./LLM/Lib/"+self.robot_name+"/prompts/"+self.abstraction+".txt"
        self.env_prompt_path="./LLM/Lib/"+self.robot_name+"/env_prompts/basic.txt"


        self.inits=OrderedDict()
        self.inits["from Lib."+self.robot_name+".FunctionLibrary import FunctionLib"] = -1
        self.inits["import rclpy"] = -1
        self.inits["rclpy.init()"] = -1
        self.inits["node = rclpy.create_node(\'gpt\')"] = -1
        self.inits["lib = FunctionLib()"] = -1

        with open(self.sysprompt_path, "r") as f:
            self.sysprompt = f.read()

        with open(self.prompt_path, "r") as f:
            self.prompt = f.read()

        with open(self.env_prompt_path, "r") as f:
            self.env_prompt = f.read()

        self.code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)

        self.chat_history = [
            {
                "role": "system",
                "content": self.sysprompt
            },
            {
                "role": "user",
                "content": self.prompt + " \n Move robot arm up by 5cm."
            },
            {
                "role": "assistant",
                "content": """```python
from Lib."""+self.robot_name+""".FunctionLibrary import FunctionLib
import rclpy

# Create rclpy node called gpt
rclpy.init()
node = rclpy.create_node(\'gpt\')

# Initialize function library
lib = FunctionLib()

# Get gripper current pose
gripper_pose = lib.get_gripper_pose()

# Add 0.05 meters to Z axis of current pose
gripper_pose[2] +=0.05

# Move up by 0.05 meters 
lib.move_gripper(gripper_pose)

print("Task finished")
```
This code gets the current pose of the gripper and moves it up by 0.05 meters."""
            }
        ]

        self.init_history = copy.deepcopy(self.chat_history)
        msg = String()
        msg.data ="Welcome to the "+ self.robot_name +" chatbot! I am ready to help you with your "+ self.robot_name +" questions and commands."
        self.responsePublisher.publish(msg)
        self.logger.info("Done")


    def ask(self, prompt):
        self.chat_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.chat_history
        )
        self.chat_history.append(
            {
                "role": "assistant",
                "content": completion.choices[0].message.content,
            }
        )
        return self.chat_history[-1]["content"]

    def extract_python_code(self, content):
        code_blocks = self.code_block_regex.findall(content)
        if code_blocks:
            full_code = "\n".join(code_blocks)

            if full_code.startswith("python"):
                full_code = full_code[7:]

            return full_code
        else:
            return None
        
    def extract_text(self, content):
        full_text = self.code_block_regex.sub('',content)
        return full_text.strip()
    
    def reduce_history(self):
        chat = copy.deepcopy(self.chat_history)
        chat[1:3]=copy.deepcopy(self.chat_history[-3:-1])
        
        self.logger.info("History reduced")
        self.chat_history = copy.deepcopy(chat[0:5])

    def reset_history(self):
        self.logger.info("GPT History Resetted")
        self.chat_history = copy.deepcopy(self.init_history)
        self.reset_count+=1

    def add_grounding_to_prompt(self,question):
        if "function" in question.lower() or "generic" in question.lower() or "code" in question.lower():
            question+= " If you wrote a function, remember to add a example function call at the end."

        question="By using the function library you are provided. "+question
        question+=" make sure to move back to home after the task is finished."

        return question

    def verify_code(self,file_path): 
        with open(file_path, "r") as file:   
            line_number = 0
            data = file.readlines()
            for line in data:
                for key in self.inits.keys():
                    if key in line and not "#" in line:#key == line[:-1]:
                        self.inits[key]=line_number
                
                line_number += 1
        
            key_number = 1
            for key in self.inits.keys():
                val = self.inits[key]
                if val == -1:
                    data.insert(key_number,key+"\n")
                    key_number += 1
                    self.inits[key]=key_number
                else: 
                    key_number = val+1

        rospy_line_num = self.inits["rclpy.init()"]
        rospy_line = data[rospy_line_num]
        lib_line_num = self.inits["lib = FunctionLib()"]
        lib_line = data[lib_line_num]

        if lib_line_num < rospy_line_num:
            data[lib_line_num] = rospy_line
            data[rospy_line_num] = lib_line

        with open(file_path, "w") as file:
            file.writelines(data)

    def get_gpt_response(self,question):

        question = self.add_grounding_to_prompt(question)
        try:
            response = self.ask(question)
        except:
            self.logger.info("Exceeded number of tokens. Code not generated")
            self.reset_history()
            return False
        
        self.reset_count = 0

        f = open("./LLM/gpt_code.py", "w")
        h = open('./LLM/chat_history.txt', 'a')
        h.write(question)
        h.write(response)
            
        code = self.extract_python_code(response)

        text = self.extract_text(response)
        msg = String()
        msg.data = text

        if code is not None:
            self.responsePublisher.publish(msg)
            f.write(code)
            f.close()
            # self.verify_code("./LLM/gpt_code.py")
            try:
                self.logger.info("running code...")
                msg = String()
                msg.data = "True"
                self.execPublisher.publish(msg)
            except Exception as e:
                msg = String()
                msg.data = "An exception occured while running the code!\n"
                self.responsePublisher.publish(msg)
                self.logger.info("exception occured while running code")
                print(e)
            else:
                msg = String()
                msg.data = "Ready!"
                self.responsePublisher.publish(msg)
                self.logger.info("Ready for running the code")

            h.close()

            return True
        else:

            h.close()

            return False

if __name__ == '__main__':
    rclpy.init(args=sys.argv)
    files=1
    reset=True
    gpt=GPT()
    
    while rclpy.ok():
        flag=False

        _,question = wait_for_message(String, gpt.node, "/llm_propmt")
        question = question.data

        if question == "!quit" or question == "!exit":
            break
        
        if "save" in question:
            shutil.copyfile('./LLM/gpt_code.py', "./LLM/gpt_code_"+str(files)+".py")
            gpt.responsePublisher.publish("Code saved with number "+str(files))
            files+=1
            flag = True

        if question == "reset":
            gpt.responsePublisher.publish("Resetting GPT! Last saved file number: "+str(files-1))
            gpt.reset_history()
            flag = True
            
        while not flag:
            flag=gpt.get_gpt_response(question)
            gpt.logger.info("in the loop.")
            if gpt.reset_count > 1:
                break

    gpt.logger.info("LLM Node killed.")
    rclpy.shutdown()