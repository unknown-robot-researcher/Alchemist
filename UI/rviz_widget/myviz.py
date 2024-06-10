#!/usr/bin/env python

# import roslib; #roslib.load_manifest('rviz_python_tutorial')
import rclpy
import sys

## Next import all the Qt bindings into the current namespace, for
## convenience.  This uses the "python_qt_binding" package which hides
## differences between PyQt and PySide, and works if at least one of
## the two is installed.  The RViz Python bindings use
## python_qt_binding internally, so you should use it here as well.
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

#from rviz import bindings as rviz
import rviz

class MyViz( QWidget ):
    def __init__(self,file):
        QWidget.__init__(self)

        self.frame = rviz.VisualizationFrame()
        self.frame.setSplashPath( "" )
        self.frame.initialize()

        reader = rviz.YamlConfigReader()
        config = rviz.Config()
        reader.readFile( config, "./UI/rviz_widget/confs/"+file.lower()+"_config.rviz" ) # config.myviz rviz_configs.rviz
        self.frame.load( config )

        self.setWindowTitle( config.mapGetChild( "Title" ).getValue() )

        self.frame.setMenuBar( None )
        self.frame.setStatusBar( None )
        self.frame.setHideButtonVisibility( False )

        self.manager = self.frame.getManager()

        self.grid_display = self.manager.getRootDisplayGroup().getDisplayAt( 0 )

        layout = QVBoxLayout()
        layout.addWidget( self.frame )
        
        self.setLayout( layout )