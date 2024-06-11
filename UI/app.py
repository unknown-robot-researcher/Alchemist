# importing required libraries
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
# from UI.rviz_widget.myviz import MyViz
import os, time
import sys
import rclpy
from std_msgs.msg import String 
from std_msgs.msg import Bool
from UI.codeEditor.higlight import PythonHighlighter
from PyQt5.QtWidgets import QWidget
from threading import Thread

os.environ["QT_API"] = "pyqt5"
from pyqtconsole.console import PythonConsole
import subprocess

def help():
	help_txt = """
run(file_number) : Runs the code generated by GPT. if you are running the last code generated by GPT file_number is 0 otherwise the appropriate file number.

setup() : Clears the entire workspace and sets basic setup which includes tables, wall and the robot. 

home() : Moves the robot back to home positions. If the robot had an object grasped first opens the gripper and then moves back home.

show(file_number) : Shows the function in the file and how to use it.

func(file_number,[inputs]): runs the function in the file with specified inputs. Check function with show before runnning. 

save() : Saves the last generated code by GPT.

reset(): Resets GPT.
				"""
	
	print(help_txt)

def run_gpt_code(num):
	print("Running GPT Code")
	if num == 0:
		try:
			result = subprocess.check_output(["python","./LLM/gpt_code.py"] , stderr=subprocess.STDOUT)
		except subprocess.CalledProcessError as e:
			print(suggestions(e.output))
			# print("Error occured while running the code. You can consult the administrators")
			return
	else:
		try:
			result = subprocess.check_output(["python","./LLM/gpt_code_"+str(num)+".py"] , stderr=subprocess.STDOUT)
		except subprocess.CalledProcessError as e:
			print(suggestions(e.output))
			# print("Error occured while running the code. You can consult the administrators")
			return

	result = parse_output(result)
	print(result)

def setup():
	print("Setting up the basic environment...")
	result = subprocess.check_output(["python","./LLM/setup_env.py"])
	result = parse_output(result)
	print(result)
	print("Done")

def home():
	print("Moving the robot back to home position...")
	result = subprocess.check_output(["python","./LLM/move_to_home.py"])
	result = parse_output(result)
	print(result)
	print("Done")

def show_function_in_file(num):
	print("Checking file...")
	if num == 0:
		found,function_sign,function_call,_ = check_code_for_function("./LLM/gpt_code.py")
		if found:
			print("Your functions signature is:")
			print(function_sign)
			print("Example function call is:")
			print(function_call)
			print("To run the function in this terminal use this:")
			input_names = function_sign.split("(")[1]
			print("func("+str(num)+",["+input_names[:-1]+"])")
		else:
			print("No functions are found in this file")
	else:
		found,function_sign,function_call,_ = check_code_for_function("./LLM/gpt_code_"+str(num)+".py")
		if found:
			print("Your functions signature is:")
			print(function_sign)
			print("Example function call is:")
			print(function_call)
			print("To run the function in this terminal use this:")
			input_names = function_sign.split("(")[1]
			print("func("+str(num)+",["+input_names[:-1]+"])")
		else:
			print("No functions are found in this file")

def run_gpt_function(num,inputs):
	print("Running custom function...")
	if num == 0:
		found,function_sign,function_call,function_call_line = check_code_for_function("./LLM/gpt_code.py")
		if found:
			print("Your functions signature is:")
			print(function_sign)
			print("Example function call is:")
			print(function_call)
			if function_call == "":
				change_function_call("./LLM/gpt_code.py",function_call_line,inputs,func_sign=function_sign)
			else:
				change_function_call("./LLM/gpt_code.py",function_call_line,inputs)
			result = subprocess.check_output(["python","./LLM/gpt_code.py"])
		else:
			print("No functions are found in this file.")
	else:
		found,function_sign,function_call,function_call_line = check_code_for_function("./LLM/gpt_code_"+str(num)+".py")
		if found:
			print("Your functions signature is:")
			print(function_sign)
			print("Example function call is:")
			print(function_call)
			if function_call == "":
				change_function_call("./LLM/gpt_code_"+str(num)+".py",function_call_line,inputs,func_sign=function_sign)
			else:
				change_function_call("./LLM/gpt_code_"+str(num)+".py",function_call_line,inputs)
			result = subprocess.check_output(["python","./LLM/gpt_code_"+str(num)+".py"])
		else:
			print("No functions are found in this file.")
	if found:
		result = parse_output(result)
		print(result)
		print("Done")


def suggestions(result):
	if "ROSInitException" in result:
		result += "\n There is an ROS initialization error in the code. Try adding 'rospy.init_node(\'gpt\')' before lib = Functionlib()"
	if "ImportError" in result:
		result += "\n There is an import error in the code. Please check if all necessary libraries were properly imported."
	return result

def parse_output(result):
	text = ""
	lines = result.splitlines()
	for line in lines:
		tokens = line.split(":")
		if len(tokens) == 3:
			text+= tokens[1]+tokens[2][:-4]+"\n"
		elif len(tokens) > 1:
			text+= tokens[1][:-4]+"\n"
		else:
			text+= line+"\n"
	return text[:-1]

def check_code_for_function(file_path):
    function_sign=""
    function_call=""
    function_name=None
    found = False
    function_call_line = 0
    line_number = 0
    with open(file_path, "r") as file:   
        data = file.readlines()
        for line in data:
            if not found:
                if "def" in line:
                    function_sign+=line[4:-2]
                    tokens=function_sign.split("(")
                    function_name = tokens[0]
                    found = True
            else:
                if function_name in line and not "#" in line:
                    function_call+=line[0:-1]
                    function_call_line = line_number
            line_number+=1
	
        if function_call_line == 0:
            function_call_line = len(data)+1

    return found,function_sign,function_call,function_call_line

def change_function_call(file_path,fcall_line,inputs,func_sign=""):
	new_call = ""
	with open(file_path, "r") as file:   
		data = file.readlines()
		if fcall_line == len(data)+1:
			old_call = func_sign
		else:
			old_call = data[fcall_line]
		new_call+=old_call[0:old_call.find("(")]
		new_call+="("
		for input in inputs:
			if isinstance(input, int) or isinstance(input, float):
				new_call+= str(input) + ","
			else:	
				new_call+= "\""+ input + "\","
		new_call = new_call[:-1] + ")\n"

		if fcall_line == len(data)+1:
			data.append(new_call)
		else:
			data[fcall_line]=new_call
	with open(file_path, "w") as file:
		file.writelines(data)


class startUpWindow(QMainWindow):
	def __init__(self):
		super(startUpWindow,self).__init__()
		layout = QVBoxLayout()
		self.combobox1 = QComboBox()
		self.combobox1.addItems(['xArm7','Panda', 'UR5', 'TIAGo'])
		label1 = QLabel("Select Robot Platform")
		self.combobox2 = QComboBox()
		self.combobox2.addItems(['Low', 'High'])
		label2 = QLabel("Select Abstraction Level")
		layout.addWidget(label1)
		layout.addWidget(self.combobox1)
		layout.addWidget(label2)
		layout.addWidget(self.combobox2)
		button = QPushButton("Start")
		button.clicked.connect(self.showMainWindow)
		layout.addWidget(button)
		container = QWidget()

		# setting layout to the container
		container.setLayout(layout)

		# Center Screen
		qtRectangle = self.frameGeometry()
		centerPoint = QDesktopWidget().availableGeometry().center()
		qtRectangle.moveCenter(centerPoint)
		self.move(qtRectangle.topLeft())

		# making container as central widget
		self.setCentralWidget(container)
		
	def showMainWindow(self):
		global window, robot, abstractionLevel
		robot = self.combobox1.currentText()
		abstractionLevel = self.combobox2.currentText()
		window = MainWindow()
		window.node.get_logger().info("Selected Robot: "+robot+" and abstraction level: "+abstractionLevel)
		window.timer.start(100)
		window.hide_show()
		# window.hide_show_tree()


class MainWindow(QMainWindow):

	def __init__(self,*args, **kwargs):
		super(MainWindow, self).__init__(*args, **kwargs)

		#GUI node
		self.node = rclpy.create_node('GUI')
		self.resp_sub = self.node.create_subscription(String, "/llm_response" ,self.store_llm_response,1)

		self.test_sub = self.node.create_subscription(String, "/test" ,self.test_topic,1)
		
		
		self.run_sub = self.node.create_subscription(String, "/run_gpt_code", self.store_run_gpt_code,10)
		self.sp2t_sub = self.node.create_subscription(String, "/speech_text", self.store_user_speech,10)
		self.chatPublisher=self.node.create_publisher(String, "/llm_propmt",10)
		self.robot_name_pub=self.node.create_publisher(String, "/start_llm",1)
		self.abstraction_pub=self.node.create_publisher(String, "/abstraction",1)
		self.whisper_pub=self.node.create_publisher(String, "/whisper",1)

		self.llm_responses=[]
		self.user_speech=[]
		self.last_displayed_response=0
		self.last_displayed_speech=0
		self.updated=False
		self.record_updated=False
		self.timer = QTimer()
		self.timer.timeout.connect(self.display_llm_response)
		self.timer.timeout.connect(self.display_record_content)
		self.timer.start(1000)

		# Center Screen
		qtRectangle = self.frameGeometry()
		centerPoint = QDesktopWidget().availableGeometry().center()
		qtRectangle.moveCenter(centerPoint)
		self.move(qtRectangle.topLeft())

		layout1 = QVBoxLayout()
		layout2 = QHBoxLayout()
		self.editor = QPlainTextEdit()
		fixedfont = QFontDatabase.systemFont(QFontDatabase.FixedFont)
		fixedfont.setPointSize(12)
		self.editor.setFont(fixedfont)

		self.tree = QTreeView()
		self.model = QFileSystemModel(self.tree)
		self.root = self.model.setRootPath('./Alchemist/LLM')
		self.model.setNameFilters(["gpt_code*"])
		self.model.setNameFilterDisables(False)  # Set to True to disable the filtering
		self.model.setFilter(QDir.NoDotAndDotDot |  QDir.Files)
		self.tree.setModel(self.model)
		self.tree.setRootIndex(self.root)
		self.tree.setAnimated(False)
		self.tree.setIndentation(10)
		self.tree.setWindowTitle("Dir View")
		self.tree.setSortingEnabled(False)
		self.tree.resize(10, 480)
		self.tree.clicked.connect(self.tree_file_open)
        # Hide size, type, and date modified columns
		self.tree.setColumnHidden(1, True)  # Size column
		self.tree.setColumnHidden(2, True)  # Type column
		self.tree.setColumnHidden(3, True)  # Date modified column
		self.tree.setHeaderHidden(True)

		# self.path holds the path of the currently open file.
		# If none, we haven't got a file open yet (or creating new).
		self.path = "./LLM/gpt_code.py"
		speech_button = QPushButton(self)
		speech_button.setStatusTip("speech to text")
		speech_button.setCheckable(True)
		speech_button.setIcon(QIcon(QPixmap("./Alchemist/UI/icons/mic.png")))
		# speech_button.setIconVisibleInMenu(True)
		speech_button.clicked.connect(self.speech_record)

		# for recording user speech
		self.record_pressed = True

		self.llm_label = QLabel("LLM is typing...")
		self.llm_label.hide()

		self.text_area = QTextEdit()
		self.text_area.setReadOnly(True)
		self.text_area.setFocusPolicy(Qt.ClickFocus)
		
		self.message = QLineEdit()
		input_layout = QHBoxLayout()
		input_layout.addWidget(self.message)
		input_layout.addWidget(speech_button)

		chatBox = QVBoxLayout()
		chatBox.addWidget(self.text_area)
		chatBox.addWidget(self.llm_label)
		chatBox.addLayout(input_layout)
        
		self.message.returnPressed.connect(self.send_message)

		self.redColor = QColor(255, 0, 0)
		self.blackColor = QColor(0, 0, 0)
		self.greenColor = QColor(0, 255, 0)

		# adding editor, chatbox and rviz to the layout
		self.__highlighter = PythonHighlighter(self.editor.document())
		# layout2.addWidget(self.editor)

		# self.rviz=MyViz(robot)

		self.console = PythonConsole()
		self.console.push_local_ns('run', run_gpt_code)
		self.console.push_local_ns('setup', setup)
		self.console.push_local_ns('home', home)
		self.console.push_local_ns('show',show_function_in_file)
		self.console.push_local_ns('func',run_gpt_function)
		self.console.push_local_ns('save',self.save_gpt_code)
		self.console.push_local_ns('reset',self.reset_gpt)
		self.console.push_local_ns('help',help)

		lower_box = QHBoxLayout()

		chat_container = QWidget()
		chat_container.setLayout(chatBox)
		chat_console_splitter = QSplitter(Qt.Horizontal)
		chat_console_splitter.addWidget(chat_container)
		chat_console_splitter.addWidget(self.console)

		lower_box.addWidget(chat_console_splitter)

		editor_rviz_splitter = QSplitter(Qt.Horizontal)

		editor_rviz_splitter.addWidget(self.tree)
		editor_rviz_splitter.addWidget(self.editor)
		# editor_rviz_splitter.addWidget(self.rviz)
		layout2.addWidget(editor_rviz_splitter)
		editor_rviz_splitter.setStretchFactor(1, 2)

		top_container = QWidget()
		top_container.setLayout(layout2)

		low_container = QWidget()
		low_container.setLayout(lower_box)

		vertical_splitter=QSplitter(Qt.Vertical)

		vertical_splitter.addWidget(top_container)
		vertical_splitter.addWidget(low_container)

		layout1.addWidget(vertical_splitter)
		# layout1.addLayout(layout2)

		# layout1.addLayout(lower_box)

		# creating a QWidget layout
		container = QWidget()

		# setting layout to the container
		container.setLayout(layout1)

		# making container as central widget
		self.setCentralWidget(container)

		# creating a status bar object
		self.status = QStatusBar()

		# setting stats bar to the window
		self.setStatusBar(self.status)

		# creating another tool bar for editing text
		view_toolbar = QToolBar("View")

		# adding this tool bar to the main window
		self.addToolBar(view_toolbar)

		# for toggling the directory tree
		tree_toggle_action = QAction("File Tree", self)
		tree_toggle_action.setStatusTip("Toggle to hide or show directory tree")
		tree_toggle_action.triggered.connect(self.hide_show_tree)
		# view_menu.addAction(toggle_action)
		view_toolbar.addAction(tree_toggle_action)

		# for toggling the editor
		toggle_action = QAction("Editor", self)
		toggle_action.setStatusTip("Toggle to hide or show editor window")
		toggle_action.triggered.connect(self.hide_show)
		# view_menu.addAction(toggle_action)
		view_toolbar.addAction(toggle_action)

		# creating a file tool bar
		file_toolbar = QToolBar("File")

		# adding file tool bar to the window
		self.addToolBar(file_toolbar)

		# creating a file menu
		file_menu = self.menuBar().addMenu("&File")

		# creating actions to add in the file menu
		# creating a open file action
		open_file_action = QAction("Open file", self)

		# setting status tip
		open_file_action.setStatusTip("Open file")

		# adding action to the open file
		open_file_action.triggered.connect(self.file_open)

		# adding this to file menu
		file_menu.addAction(open_file_action)

		# adding this to tool bar
		file_toolbar.addAction(open_file_action)

		# similarly creating a save action
		save_file_action = QAction("Save", self)
		save_file_action.setStatusTip("Save current page")
		save_file_action.triggered.connect(self.file_save)
		file_menu.addAction(save_file_action)
		file_toolbar.addAction(save_file_action)

		# [ctrl + s] shortcut to save file 
		shortcut = QShortcut(QKeySequence.Save, self)
		shortcut.activated.connect(self.file_save)

		# similarly creating save action
		saveas_file_action = QAction("Save As", self)
		saveas_file_action.setStatusTip("Save current page to specified file")
		saveas_file_action.triggered.connect(self.file_saveas)
		file_menu.addAction(saveas_file_action)
		file_toolbar.addAction(saveas_file_action)

		# for print action
		print_action = QAction("Print", self)
		print_action.setStatusTip("Print current page")
		print_action.triggered.connect(self.file_print)
		file_menu.addAction(print_action)
		file_toolbar.addAction(print_action)

		# creating another tool bar for editing text
		edit_toolbar = QToolBar("Edit")

		# adding this tool bar to the main window
		self.addToolBar(edit_toolbar)

		# creating a edit menu bar
		edit_menu = self.menuBar().addMenu("&Edit")

		# adding actions to the tool bar and menu bar

		# undo action
		undo_action = QAction("Undo", self)
		# adding status tip
		undo_action.setStatusTip("Undo last change")

		# when triggered undo the editor
		undo_action.triggered.connect(self.editor.undo)

		# adding this to tool and menu bar
		edit_toolbar.addAction(undo_action)
		edit_menu.addAction(undo_action)

		# redo action
		redo_action = QAction("Redo", self)
		redo_action.setStatusTip("Redo last change")

		# when triggered redo the editor
		redo_action.triggered.connect(self.editor.redo)

		# adding this to menu and tool bar
		edit_toolbar.addAction(redo_action)
		edit_menu.addAction(redo_action)

		# cut action
		cut_action = QAction("Cut", self)
		cut_action.setStatusTip("Cut selected text")

		# when triggered cut the editor text
		cut_action.triggered.connect(self.editor.cut)

		# adding this to menu and tool bar
		edit_toolbar.addAction(cut_action)
		edit_menu.addAction(cut_action)

		# copy action
		copy_action = QAction("Copy", self)
		copy_action.setStatusTip("Copy selected text")

		# when triggered copy the editor text
		copy_action.triggered.connect(self.editor.copy)

		# adding this to menu and tool bar
		edit_toolbar.addAction(copy_action)
		edit_menu.addAction(copy_action)

		# paste action
		paste_action = QAction("Paste", self)
		paste_action.setStatusTip("Paste from clipboard")

		# when triggered paste the copied text
		paste_action.triggered.connect(self.editor.paste)

		# adding this to menu and tool bar
		edit_toolbar.addAction(paste_action)
		edit_menu.addAction(paste_action)

		# select all action
		select_action = QAction("Select all", self)
		select_action.setStatusTip("Select all text")

		# when this triggered select the whole text
		select_action.triggered.connect(self.editor.selectAll)

		# adding this to menu and tool bar
		edit_toolbar.addAction(select_action)
		edit_menu.addAction(select_action)


		# wrap action
		wrap_action = QAction("Wrap text to window", self)
		wrap_action.setStatusTip("Check to wrap text to window")

		# making it checkable
		wrap_action.setCheckable(True)

		# making it checked
		wrap_action.setChecked(True)

		# adding action
		wrap_action.triggered.connect(self.edit_toggle_wrap)

		# adding it to edit menu not to the tool bar
		edit_menu.addAction(wrap_action)

		# calling update title method
		self.update_title()

		# showing all the components
		self.show()

		self.console.eval_in_thread()
		#self.console.eval_queued()
		self.run_flag=False
		msg = String()
		msg.data = robot
		self.robot_name_pub.publish(msg)
		time.sleep(1)
		msg.data = abstractionLevel
		self.abstraction_pub.publish(msg)

		# The status of the editor (not hidden)
		self.hidden = False
		self.tree_hidden = False

	# creating dialog critical method
	# to show errors
	def dialog_critical(self, s):

		# creating a QMessageBox object
		dlg = QMessageBox(self)

		# setting text to the dlg
		dlg.setText(s)

		# setting icon to it
		dlg.setIcon(QMessageBox.Critical)

		# showing it
		dlg.show()

	# action called by file open action
	def file_open(self):

		# getting path and bool value
		path, _ = QFileDialog.getOpenFileName(self, "Open file", "",
							"Text documents (*.txt);All files (*.*)")

		# if path is true
		if path:
			# try opening path
			try:
				with open(path, 'rU') as f:
					# read the file
					text = f.read()

			# if some error occurred
			except Exception as e:

				# show error using critical method
				self.dialog_critical(str(e))
			# else
			else:
				# update path value
				self.path = path

				# update the text
				self.editor.setPlainText(text)

				# update the title
				self.update_title()

    # Custom slot to open file in the QPlainTextEdit
	def tree_file_open(self, index):
		file_path = self.model.filePath(index)
		self.path = file_path # Have to do this in order to be able to save
		if self.model.isDir(index):  # Open only if it's a file
			return

		with open(file_path, 'r') as file:
			file_content = file.read()
			self.editor.setPlainText(file_content)


	# action called by file save action
	def file_save(self):

		# if there is no save path
		if self.path is None:

			# call save as method
			return self.file_saveas()

		# else call save to path method
		self._save_to_path(self.path)

	# action called by save as action
	def file_saveas(self):

		# opening path
		path, _ = QFileDialog.getSaveFileName(self, "Save file", "",
							"Text documents (*.txt);All files (*.*)")

		# if dialog is cancelled i.e no path is selected
		if not path:
			# return this method
			# i.e no action performed
			return

		# else call save to path method
		self._save_to_path(path)

	# save to path method
	def _save_to_path(self, path):

		# get the text
		text = self.editor.toPlainText()

		# try catch block
		try:

			# opening file to write
			with open(path, 'w') as f:

				# write text in the file
				f.write(text)

		# if error occurs
		except Exception as e:

			# show error using critical
			self.dialog_critical(str(e))

		# else do this
		else:
			# change path
			self.path = path
			# update the title
			self.update_title()

	# action called by print
	def file_print(self):

		# creating a QPrintDialog
		dlg = QPrintDialog()

		# if executed
		if dlg.exec_():

			# print the text
			self.editor.print_(dlg.printer())

	def hide_show(self):
		if self.hidden:
			self.editor.show()
			self.hidden = False
			self.tree.show()	
			self.tree_hidden = False

		else:
			self.editor.hide()
			self.hidden = True
			self.tree.hide()
			self.tree_hidden = True

	def hide_show_tree(self):
		if self.tree_hidden:
			self.tree.show()
			self.tree_hidden = False
		else:
			self.tree.hide()
			self.tree_hidden = True



	def speech_record(self):
		msg = String()
		if self.record_pressed:
			self.record_pressed = False
			msg.data = "True"
			self.whisper_pub.publish(msg)
		else:
			self.record_pressed = True
			msg.data = "False"
			self.whisper_pub.publish(msg)



	# update title method
	def update_title(self):

		# setting window title with prefix as file name
		# suffix as PyQt5 Notepad
		self.setWindowTitle("%s - Natural Robot" %(os.path.basename(self.path)
												if self.path else "Untitled"))

	# action called by edit toggle
	def edit_toggle_wrap(self):

		# chaining line wrap mode
		self.editor.setLineWrapMode(1 if self.editor.lineWrapMode() == 0 else 0 )

	# chatbox send message to llm 
	def send_message(self):
		prompt=self.message.text()
		self.node.get_logger().info(prompt)
		msg = String()
		msg.data = prompt
		self.chatPublisher.publish(msg)
		self.message.clear()
		self.text_area.moveCursor(QTextCursor.End)
		self.text_area.setTextColor(self.redColor)
		self.text_area.append("User> ")
		self.text_area.setTextColor(self.blackColor)
		self.text_area.insertPlainText(prompt)
		#self.gpt.get_gpt_response(prompt)
		self.llm_label.show() # show llm typing

	def save_gpt_code(self):
		print("Saving code...")
		msg = String()
		msg.data = "save"
		self.chatPublisher.publish(msg)

	def reset_gpt(self):
		print("Resetting LLM...")
		msg = String()
		msg.data = "reset"
		self.chatPublisher.publish(msg)

	def store_llm_response(self,data):
		new_response=data.data
		print(new_response)
		self.llm_responses.append(new_response)
		self.updated=True

	def test_topic(self,msg):
		print(msg.data)

	def test_service(self,msg):
		print("service works")

	def store_user_speech(self,data):
		new_response=data.data
		self.user_speech.append(new_response)
		self.record_updated=True

	def store_run_gpt_code(self,data):
		self.run_flag=data.data=="True"
		#print("did this")

	# chatbox recieve and display text from llm
	def display_llm_response(self):
		if self.updated or len(self.llm_responses)>self.last_displayed_response:
			self.llm_label.hide() # hiding llm writing
			new_response=self.llm_responses[self.last_displayed_response]
			self.text_area.moveCursor(QTextCursor.End)
			self.text_area.setTextColor(self.greenColor)
			self.text_area.append(robot+"> ")
			self.text_area.setTextColor(self.blackColor)
			self.text_area.insertPlainText(new_response)
			self.show_llm_code()
			self.last_displayed_response+=1
			self.updated=False
			if self.run_flag:
				self.console.insert_input_text('run(0)')
				#self.console.enterEvent(QEvent(QEvent.KeyPress)).emit()
				# cursor = self.console._textCursor()
				# cursor.movePosition(QTextCursor.End)
				# self.console._setTextCursor(cursor)
				# buffer = self.console.input_buffer()
				# self.console._hide_cursor()
				# self.console.insert_input_text('\n', show_ps=False)
				# self.console.process_input(buffer)
				#self.console._handle_enter_key(QEvent.KeyPress)
				self.run_flag=False
				#print('made false')
		
	def display_record_content(self):
		if self.record_updated or len(self.user_speech)>self.last_displayed_speech:
			new_response=self.user_speech[self.last_displayed_speech]
			self.node.get_logger().info(new_response)
			msg = String()
			msg.data = new_response
			self.chatPublisher.publish(msg)
			self.text_area.moveCursor(QTextCursor.End)
			self.text_area.setTextColor(self.redColor)
			self.text_area.append("User> ")
			self.text_area.setTextColor(self.blackColor)
			self.text_area.insertPlainText(new_response)
			self.last_displayed_speech+=1
			self.record_updated=False


	def show_llm_code(self):
		try:
			with open(self.path, 'rU') as f:
				# read the file
				text = f.read()

		# if some error occurred
		except Exception as e:

			# show error using critical method
			self.dialog_critical(str(e))
		# else
		else:
			# update the text
			self.editor.setPlainText(text)

			# update the title
			self.update_title()

def closing_llm():
	node = rclpy.create_node('llm_killer')
	msg = String()
	msg.data = "!quit"
	p = node.create_publisher(String, "/llm_propmt",10)
	time.sleep(1)
	p.publish(msg)

def closing_whisper():
	node = rclpy.create_node('whisper_killer')
	msg = String()
	msg.data = "!quit"
	p = node.create_publisher(String, "/whisper",10)
	time.sleep(1)
	p.publish(msg)

# drivers code
#if __name__ == '__main__':

def Main():
	rclpy.init()
	global window

	# creating PyQt5 application
	app = QApplication(sys.argv)

	# setting application name
	app.setApplicationName("Alchemist")
	app.aboutToQuit.connect(closing_llm)
	app.aboutToQuit.connect(closing_whisper)


	# creating a main window object
	#window = MainWindow()
	window = startUpWindow()
	window.show()

	# loop
	sys.exit(app.exec_())